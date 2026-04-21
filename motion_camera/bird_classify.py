"""
bird_classify.py - TensorFlow Lite inference with multi-frame voting.

Public API
----------
  classifier = BirdClassifier()          # load model once at startup
  result = classifier.vote(frames)       # classify a list of burst frames
  result = classifier.classify_frame(f)  # classify a single frame (testing)

VoteResult is a named tuple:
  label       str    - winning label string from labels.txt
  confidence  float  - average confidence across winning votes (0-1)
  top_3       list   - top-3 [(label, confidence), …] from the best single frame
  vote_count  int    - how many frames voted for the winning label
  frame_count int    - how many frames were actually classified
"""

import warnings
from collections import defaultdict
from typing import NamedTuple

import cv2
import numpy as np

import config
import motion_detect

# Suppress noisy NumPy subnormal warning that appears on Pi with TFLite
warnings.filterwarnings(
    "ignore",
    message="The value of the smallest subnormal for <class 'numpy.float64'> type is zero.",
    category=UserWarning,
    module="numpy.core.getlimits",
)

# CLAHE = Contrast Limited Adaptive Histogram Equalization, a local contrast enhancement technique
# Build CLAHE instance once at import time - cv2.createCLAHE is not free.
_clahe = cv2.createCLAHE(
    clipLimit=config.CLAHE_CLIP_LIMIT,
    tileGridSize=config.CLAHE_TILE_SIZE,
)

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class VoteResult(NamedTuple):
    label:       str
    confidence:  float
    top_3:       list   # [(label_str, confidence_float), ...]
    vote_count:  int    # frames that voted for the winning label
    frame_count: int    # total frames that passed texture check

# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def _load_labels(path: str) -> list[str]:
    try:
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return []


def _parse_names(label: str) -> tuple[str, str]:
    """
    Split a label like 'Cardinalis cardinalis (Northern Cardinal)'
    into (scientific, common).  Falls back gracefully if the format differs.
    """
    if "(" in label and label.endswith(")"):
        paren = label.index("(")
        scientific = label[:paren].strip()
        common     = label[paren + 1:-1].strip()
        return scientific, common
    return label, label


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class BirdClassifier:
    """
    Wraps a TFLite interpreter and exposes vote() for burst classification.
    """

    def __init__(self) -> None:
        self.labels: list[str] = _load_labels(config.LABELS_PATH)
        self.interpreter = self._load_interpreter()
        self.interpreter.allocate_tensors()
        self._inp  = self.interpreter.get_input_details()
        self._out  = self.interpreter.get_output_details()
        self._h    = self._inp[0]["shape"][1]
        self._w    = self._inp[0]["shape"][2]
        self._fp32 = self._inp[0]["dtype"] == np.float32

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_interpreter():
        try:
            import tflite_runtime.interpreter as tflite
            return tflite.Interpreter(model_path=config.MODEL_PATH)
        except ImportError:
            import tensorflow as tf
            return tf.lite.Interpreter(model_path=config.MODEL_PATH)

    @staticmethod
    def _apply_clahe(image: np.ndarray) -> np.ndarray:
        """
        Boost local contrast via CLAHE on the L channel of LAB colour space.
        This corrects colour casts and backlit conditions without touching
        hue or saturation - so the model still sees realistic bird colours,
        just with better local contrast.

        Input/output: BGR uint8 numpy array (unchanged shape).
        """
        lab            = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b        = cv2.split(lab)
        l_eq           = _clahe.apply(l)
        lab_eq         = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        CLAHE → resize → BGR→RGB → normalise → add batch dim.
        CLAHE is applied before resizing so it operates at full crop
        resolution and captures fine local contrast (feathers, edges).
        """
        enhanced = self._apply_clahe(image)
        resized  = cv2.resize(enhanced, (self._w, self._h))
        rgb      = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        if self._fp32:
            rgb = rgb.astype(np.float32) / 255.0
        return np.expand_dims(rgb, axis=0)

    def _run_inference(self, image: np.ndarray) -> np.ndarray:
        """
        Run TFLite inference on a preprocessed image.
        Returns a probability array with stable softmax applied.
        The model outputs raw logits, so we apply softmax here.
        """
        inp = self._preprocess(image)
        self.interpreter.set_tensor(self._inp[0]["index"], inp)
        self.interpreter.invoke()
        logits = self.interpreter.get_tensor(self._out[0]["index"])[0]

        # Robust softmax: handle NaN/inf, clip, stable shift
        logits  = logits.astype(np.float64)
        logits  = np.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
        logits  = np.clip(logits, -100.0, 100.0)
        shifted = logits - np.max(logits)
        exp     = np.exp(shifted)
        s       = np.sum(exp)
        if s == 0 or not np.isfinite(s):
            return np.zeros_like(exp)
        probs = exp / s
        if not np.all(np.isfinite(probs)):
            probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        return probs

    def _top3(self, probs: np.ndarray) -> list[tuple[str, float]]:
        idx = np.argsort(probs)[-3:][::-1]
        return [
            (self.labels[i] if i < len(self.labels) else f"class_{i}",
             float(probs[i]))
            for i in idx
        ]

    def _is_textured_enough(self, roi_crop: np.ndarray) -> bool:
        """Return True if the crop has enough texture to plausibly contain a bird."""
        gray    = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return lap_var >= config.LAP_VAR_THRESHOLD

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_frame(self, frame: np.ndarray) -> tuple[str, float, list] | None:
        """
        Classify a single frame (already ROI-cropped).
        Returns (label, confidence, top_3) or None if the frame is too smooth.
        Useful for testing individual frames without the full burst pipeline.
        """
        if not self._is_textured_enough(frame):
            return None
        probs      = self._run_inference(frame)
        top_idx    = int(np.argmax(probs))
        confidence = float(probs[top_idx])
        label      = self.labels[top_idx] if top_idx < len(self.labels) else f"class_{top_idx}"
        return label, confidence, self._top3(probs)

    def vote(self, frame_paths: list[str]) -> VoteResult | None:
        """
        Classify every Nth frame (CLASSIFY_EVERY_N_FRAMES) from a burst and
        aggregate the results into a single winning prediction.

        Confidence averaging fix: avg_conf is calculated only over frames that
        voted for the winning label, not over all classified frames. Dividing
        by total frame_count deflated scores when competing species appeared
        in some frames.

        Accepts file paths rather than numpy arrays so frames are loaded one
        at a time and RAM stays flat regardless of burst length.

        Args:
            frame_paths: list of JPEG paths from frame_capture.record_visit().

        Returns:
            VoteResult or None if not enough frames passed the texture check.
        """
        if not frame_paths:
            return None

        n       = config.CLASSIFY_EVERY_N_FRAMES
        sampled = frame_paths[::n]

        votes:       dict[str, float] = defaultdict(float)
        vote_counts: dict[str, int]   = defaultdict(int)   # per-label vote tally
        best_top3:   list             = []
        best_conf    = 0.0
        frame_count  = 0                                   # total frames classified

        for path in sampled:
            frame = cv2.imread(path)
            if frame is None:
                continue
            roi    = motion_detect.apply_roi(frame)
            del frame
            result = self.classify_frame(roi)
            del roi
            if result is None:
                continue
            label, conf, top3 = result
            votes[label]       += conf
            vote_counts[label] += 1   # track per-label count separately
            frame_count        += 1
            if conf > best_conf:
                best_conf = conf
                best_top3 = top3

        if frame_count < config.MIN_VOTES_REQUIRED:
            return None

        winning_label = max(votes, key=votes.__getitem__)

        agreement = vote_counts[winning_label] / frame_count
        if agreement < config.MIN_VOTE_FRACTION:
            return None

        # Divide by votes for the winner only - not all classified frames
        avg_conf = votes[winning_label] / vote_counts[winning_label]

        return VoteResult(
            label       = winning_label,
            confidence  = avg_conf,
            top_3       = best_top3,
            vote_count  = vote_counts[winning_label],  # frames that agreed
            frame_count = frame_count,                 # total frames classified
        )
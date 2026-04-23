"""
gif_builder.py - Turn on-disk JPEG frames into a GIF and a JPEG thumbnail.

Frames are loaded one at a time from disk so RAM usage stays flat regardless
of burst length.

Color correctness
-----------------
picamera2 returns correct RGB from capture_array(). Frames are written to
disk as-is (no conversion) and read back by cv2.imread() as BGR. This file
converts BGR -> RGB exactly once per frame before GIF encoding.

GIF encoding
-----------------
Uses MEDIANCUT quantization with a shared palette derived from the first
frame. All subsequent frames are quantized against that same palette via
putpalette(), preventing the per-frame channel reordering. FASTOCTREE
caused bugs, such as mis-colored GIFs (cardinals appeared blue), (sky appeared orange).

Trailing frame filter
---------------------
Frames with Laplacian variance below 60% of LAP_VAR_THRESHOLD are skipped
when building the GIF. This trims the empty-feeder tail that accumulates
during the NO_MOTION_TIMEOUT window after the bird leaves. If the filter
removes all frames (very blurry visit), it falls back to using all frames
unfiltered so the GIF never crashes.

Public API
----------
  gif_path, thumb_path = build(frame_paths, label, thumb_dir, timestamp)
"""

import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_label(label: str) -> str:
    label = label.replace(" ", "_").replace("(", "").replace(")", "")
    return "".join(c for c in label if c.isalnum() or c == "_")


def _lap_var(bgr: np.ndarray) -> float:
    """Return Laplacian variance of a BGR frame (texture measure)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 array to RGB uint8."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build(
    frame_paths: list[str],
    label: str,
    thumb_dir: str,
    timestamp: datetime | None = None,
) -> tuple[str, str]:
    """
    Save a GIF and a JPEG thumbnail from a list of on-disk JPEG frame paths.

    Args:
        frame_paths : list of paths returned by frame_capture.record_visit().
        label       : winning species label (used in filenames).
        thumb_dir   : directory for the JPEG still.
        timestamp   : datetime to embed in filenames; defaults to now.

    Returns:
        (gif_path, thumb_path) - absolute paths to the saved files.
    """
    if not frame_paths:
        raise ValueError("build() called with an empty frame list.")

    Path(config.GIFS_DIR).mkdir(parents=True, exist_ok=True)
    Path(thumb_dir).mkdir(parents=True, exist_ok=True)

    ts     = timestamp or datetime.now()
    ts_str = ts.strftime("%Y_%m_%d_%H_%M_%S")
    clean  = _clean_label(label)
    stem   = f"{ts_str}_{clean}"

    gif_path   = os.path.join(config.GIFS_DIR, f"{stem}.gif")
    thumb_path = os.path.join(thumb_dir,        f"{stem}.jpg")

    frame_duration_ms = int(1000 / config.BURST_FPS)

    # -------------------------------------------------------------------------
    # Thumbnail: sharpest frame by Laplacian variance.
    # cv2.imwrite expects BGR - frames on disk are BGR so no conversion needed.
    # -------------------------------------------------------------------------
    best_score = -1.0
    best_path  = frame_paths[0]

    for path in frame_paths:
        frame = cv2.imread(path)
        if frame is None:
            continue
        score = _lap_var(frame)
        if score > best_score:
            best_score = score
            best_path  = path
        del frame

    best_frame = cv2.imread(best_path)
    if best_frame is not None:
        cv2.imwrite(thumb_path, best_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        del best_frame

    # -------------------------------------------------------------------------
    # GIF: collect RGB frames, filtering out low-texture trailing frames.
    #
    # Threshold is 60% of LAP_VAR_THRESHOLD - loose enough to keep soft-but-
    # valid bird frames, strict enough to drop empty feeder/sky tail frames.
    # -------------------------------------------------------------------------
    gif_threshold = config.LAP_VAR_THRESHOLD * 0.6
    frames_rgb: list[np.ndarray] = []

    for path in frame_paths:
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        if _lap_var(bgr) < gif_threshold:
            del bgr
            continue
        frames_rgb.append(_bgr_to_rgb(bgr))
        del bgr

    # Fallback: if texture filter removed everything, use all frames unfiltered
    if not frames_rgb:
        for path in frame_paths:
            bgr = cv2.imread(path)
            if bgr is not None:
                frames_rgb.append(_bgr_to_rgb(bgr))
                del bgr

    if not frames_rgb:
        raise ValueError("No readable frames found - cannot build GIF.")

    # -------------------------------------------------------------------------
    # GIF encoding: MEDIANCUT with a shared palette.
    #
    # Build the palette from the first frame, then force all subsequent frames
    # to use it via putpalette(). This prevents per-frame palette reordering.
    # -------------------------------------------------------------------------
    first_pil = Image.fromarray(frames_rgb[0]).quantize(
        colors=256, method=Image.Quantize.MEDIANCUT, dither=0
    )
    palette = first_pil.getpalette()

    gif_frames = [first_pil]
    for rgb in frames_rgb[1:]:
        frame_pil = Image.fromarray(rgb).quantize(
            colors=256, method=Image.Quantize.MEDIANCUT, dither=0
        )
        frame_pil.putpalette(palette)
        gif_frames.append(frame_pil)

    gif_frames[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        append_images=gif_frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=False,
    )

    return gif_path, thumb_path
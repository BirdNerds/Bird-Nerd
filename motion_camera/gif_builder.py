"""
gif_builder.py - Turn a list of BGR frames into a GIF and a JPEG thumbnail.

Public API
----------
  gif_path, thumb_path = build(frames, label, timestamp)

  - gif_path   : path to the saved GIF  (in config.GIFS_DIR)
  - thumb_path : path to the JPEG still (in config.IMAGES_DIR or UNCLEAR_DIR
                 depending on caller - the caller chooses the directory)

The thumbnail is the sharpest frame from the burst, selected by Laplacian
variance.  It is saved separately so the website can show a static image in
the table and play the GIF only inside the detail modal.
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
    """Make label safe for use as a filename component."""
    label = label.replace(" ", "_").replace("(", "").replace(")", "")
    return "".join(c for c in label if c.isalnum() or c == "_")


def _sharpest_frame(frames: list[np.ndarray]) -> np.ndarray:
    """
    Return the frame with the highest Laplacian variance (i.e. sharpest).
    Used to pick the thumbnail so the still image looks as clean as possible.
    """
    best_score = -1.0
    best_frame = frames[0]
    for frame in frames:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if score > best_score:
            best_score = score
            best_frame = frame
    return best_frame


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """Convert a BGR numpy array to a PIL RGB Image."""
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

def build(
    frames: list[np.ndarray],
    label: str,
    thumb_dir: str,
    timestamp: datetime | None = None,
) -> tuple[str, str]:
    """
    Save a GIF and a JPEG thumbnail from a burst of BGR frames.

    Args:
        frames    : list of BGR numpy arrays at GIF resolution.
        label     : winning species label (used in filenames).
        thumb_dir : directory for the JPEG still - caller passes either
                    config.IMAGES_DIR (high-confidence) or config.UNCLEAR_DIR.
        timestamp : datetime to embed in filenames; defaults to now.

    Returns:
        (gif_path, thumb_path) - absolute paths to the saved files.

    GIF notes:
        - Frame duration is derived from config.BURST_FPS.
        - We use Pillow's LANCZOS resampler and quantize to 256 colours per
          frame (standard GIF limitation).  Pillow's default dithering is
          applied to soften colour banding.
        - loop=0 means the GIF loops forever, which suits autoplay on the site.
        - Firebase Storage handles GIFs just like any other binary blob.
    """
    if not frames:
        raise ValueError("build() called with an empty frame list.")

    Path(config.GIFS_DIR).mkdir(parents=True, exist_ok=True)
    Path(thumb_dir).mkdir(parents=True, exist_ok=True)

    ts      = timestamp or datetime.now()
    ts_str  = ts.strftime("%Y_%m_%d_%H_%M_%S")
    clean   = _clean_label(label)
    stem    = f"{ts_str}_{clean}"

    gif_path   = os.path.join(config.GIFS_DIR,  f"{stem}.gif")
    thumb_path = os.path.join(thumb_dir,         f"{stem}.jpg")

    # -- Thumbnail (sharpest frame, JPEG) ---------------------------------
    best = _sharpest_frame(frames)
    cv2.imwrite(thumb_path, best, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # -- GIF --------------------------------------------------------------
    # Frame duration in milliseconds for Pillow
    frame_duration_ms = int(1000 / config.BURST_FPS)

    pil_frames = [_bgr_to_pil(f) for f in frames]

    # Quantize each frame to a 256-colour palette (required by GIF format).
    # Using ADAPTIVE palette per-frame gives better colour accuracy than a
    # single global palette, at the cost of slightly larger files.
    quantized = [f.quantize(method=Image.Quantize.MEDIANCUT) for f in pil_frames]

    quantized[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        append_images=quantized[1:],
        duration=frame_duration_ms,
        loop=0,          # 0 = loop forever
        optimize=False,  # optimize=True can corrupt palette on multi-frame GIFs
    )

    return gif_path, thumb_path

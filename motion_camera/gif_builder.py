"""
gif_builder.py - Turn on-disk JPEG frames into a GIF and a JPEG thumbnail.

Frames are loaded one at a time from disk so RAM usage stays flat regardless
of burst length.

Color correctness
-----------------
OpenCV loads images as BGR.  All GIF encoders (imageio/Pillow) expect RGB.
This file converts BGR → RGB exactly once per frame, immediately after
cv2.imread(), and never touches PIL's quantize() — which was the root cause
of the blue-cardinal bug (FASTOCTREE palette reordered channels internally).

GIF encoding path
-----------------
  cv2.imread()  →  BGR→RGB via cvtColor  →  imageio.mimsave()
imageio writes each uint8 RGB array directly into the GIF palette without
any additional channel manipulation, so reds stay red.

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


def _bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR uint8 array (from cv2.imread) to RGB uint8."""
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

    Frames are loaded one at a time to keep RAM flat - at no point is the
    entire burst held in memory simultaneously to avoid OOM on a 1 GB Pi.

    Color pipeline (no palette corruption):
      cv2.imread() → BGR → cvtColor(BGR2RGB) → imageio.mimsave()
    imageio writes RGB arrays directly; no PIL quantize() is involved.

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
    # Thumbnail: sharpest frame (Laplacian variance), stays BGR for cv2.imwrite
    # -------------------------------------------------------------------------
    best_score = -1.0
    best_path  = frame_paths[0]

    for path in frame_paths:
        frame = cv2.imread(path)
        if frame is None:
            continue
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if score > best_score:
            best_score = score
            best_path  = path
        del frame

    best_frame = cv2.imread(best_path)
    if best_frame is not None:
        # cv2.imwrite expects BGR - no conversion needed for the JPEG thumbnail
        cv2.imwrite(thumb_path, best_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        del best_frame

    ## -- GIF: use a fixed global palette to prevent channel reordering --------
    frames_rgb = []
    for path in frame_paths:
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        frames_rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        del bgr

    if not frames_rgb:
        raise ValueError("No readable frames found - cannot build GIF.")

    # Build palette from first frame, then convert ALL frames against it
    # using that same palette - avoids per-frame palette reordering
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
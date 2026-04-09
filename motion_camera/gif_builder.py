"""
gif_builder.py - Turn on-disk JPEG frames into a GIF and a JPEG thumbnail.

Frames are loaded one at a time from disk so RAM usage stays flat regardless
of burst length.

Public API
----------
  gif_path, thumb_path = build(frame_paths, label, thumb_dir, timestamp)

No printing — main.py handles all terminal output.
"""

import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import config


def _clean_label(label: str) -> str:
    label = label.replace(" ", "_").replace("(", "").replace(")", "")
    return "".join(c for c in label if c.isalnum() or c == "_")


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


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

    # -- Thumbnail: find sharpest frame without loading all at once -------
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
        cv2.imwrite(thumb_path, best_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        del best_frame

    # -- GIF: build incrementally, one frame at a time --------------------
    first_bgr = cv2.imread(frame_paths[0])
    if first_bgr is None:
        raise ValueError(f"Could not read first frame: {frame_paths[0]}")

    first_pil = _bgr_to_pil(first_bgr).quantize(method=Image.Quantize.FASTOCTREE)
    del first_bgr

    rest: list[Image.Image] = []
    for path in frame_paths[1:]:
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        rest.append(_bgr_to_pil(bgr).quantize(method=Image.Quantize.FASTOCTREE))
        del bgr

    first_pil.save(
        gif_path,
        format="GIF",
        save_all=True,
        append_images=rest,
        duration=frame_duration_ms,
        loop=0,
        optimize=False,
    )

    return gif_path, thumb_path

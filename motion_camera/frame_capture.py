"""
frame_capture.py - Camera interface using picamera2.

Memory strategy
---------------
This program writes each frame to a temp JPEG on disk as it is captured,
then returns the list of file paths.  The caller (gif_builder.py) reads
them back one at a time, so peak RAM usage is roughly one frame at a time
rather than the entire burst.

Public API
----------
  open_camera()               -> None
  close_camera()              -> None
  grab_frame()                -> np.ndarray  (full-res BGR, for motion checks)
  record_visit(stop_event)    -> list[str]   (paths to JPEG frames on disk)
  load_frames(paths)          -> list[np.ndarray]  (read frames back for GIF)
"""

import os
import time
import threading
import numpy as np
from picamera2 import Picamera2
import cv2

import config

os.environ["LIBCAMERA_LOG_LEVELS"] = "3"

_camera: Picamera2 | None = None
# Dedicated temp directory for burst frames — wiped on each new visit
_TEMP_DIR = "/tmp/bird_nerd_burst"


def open_camera() -> None:
    """Open and warm up the camera. Call once at startup."""
    global _camera
    if _camera is not None:
        return
    os.makedirs(_TEMP_DIR, exist_ok=True)
    _camera = Picamera2()
    still_cfg = _camera.create_still_configuration(
        main={"size": (config.CAPTURE_WIDTH, config.CAPTURE_HEIGHT),
              "format": "RGB888"},
        lores={"size": (640, 480), "format": "YUV420"},
    )
    _camera.configure(still_cfg)
    _camera.start()
    time.sleep(2.0)
    _camera.set_controls({"AwbEnable": False, "AwbMode": 1})  # lock auto white balance after warmup
    # Can also try:
    # _camera.set_controls({"AwbMode": lc.AwbModeEnum.Daylight}) # Looking through a window, so daylight white balance is more consistent than auto


def close_camera() -> None:
    """Release the camera. Call on clean shutdown."""
    global _camera
    if _camera is not None:
        _camera.stop()
        _camera.close()
        _camera = None


def grab_frame() -> np.ndarray:
    """
    Capture a single full-resolution frame for motion comparison.
    Returns a BGR numpy array of shape (CAPTURE_HEIGHT, CAPTURE_WIDTH, 3).
    Always uses the still configuration - resolution is always consistent.
    """
    if _camera is None:
        raise RuntimeError("Camera not open — call open_camera() first.")
    rgb = _camera.capture_array("main")
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _clear_temp_dir() -> None:
    """Delete leftover frames from the previous visit."""
    for f in os.listdir(_TEMP_DIR):
        if f.endswith(".jpg"):
            try:
                os.remove(os.path.join(_TEMP_DIR, f))
            except OSError:
                pass


def record_visit(stop_event: threading.Event) -> list[str]:
    """
    Capture frames at BURST_FPS until stop_event is set or MAX_VISIT_DURATION
    elapses. Each frame is written to disk immediately as a JPEG so that
    RAM usage stays flat (one frame at a time) regardless of visit length.

    Returns a list of JPEG file paths in capture order.
    Peak RAM during capture: ~1 frame = 1280*960*3 bytes ~ 3.5 MB.

    Args:
        stop_event: threading.Event — set by the caller when the bird leaves.

    Returns:
        List of absolute paths to JPEG files in _TEMP_DIR.
    """
    if _camera is None:
        raise RuntimeError("Camera not open - call open_camera() first.")

    _clear_temp_dir()
    paths: list[str] = []

    interval   = 1.0 / config.BURST_FPS
    start_time = time.monotonic()
    deadline   = start_time + config.MAX_VISIT_DURATION

    # Switch to video configuration for consistent low-latency burst capture
    _camera.stop()
    video_cfg = _camera.create_video_configuration(
        main={"size": (config.GIF_WIDTH, config.GIF_HEIGHT), "format": "RGB888"},
    )
    _camera.configure(video_cfg)
    _camera.start()

    frame_idx = 0
    try:
        while not stop_event.is_set() and time.monotonic() < deadline:
            loop_start = time.monotonic()

            rgb = _camera.capture_array("main")
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Write to disk immediately - do not accumulate in RAM
            path = os.path.join(_TEMP_DIR, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            paths.append(path)
            frame_idx += 1

            elapsed   = time.monotonic() - loop_start
            remaining = interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    finally:
        # Always restore still configuration so grab_frame() works again
        _camera.stop()
        still_cfg = _camera.create_still_configuration(
            main={"size": (config.CAPTURE_WIDTH, config.CAPTURE_HEIGHT),
                  "format": "RGB888"},
            lores={"size": (640, 480), "format": "YUV420"},
        )
        _camera.configure(still_cfg)
        _camera.start()
        time.sleep(0.5)

    return paths


def load_frames(paths: list[str]) -> list[np.ndarray]:
    """
    Read JPEG frames back from disk as BGR numpy arrays.
    Used by gif_builder and bird_classify when they need raw pixel data.
    Loads one at a time — caller should not hold the entire list in memory
    longer than necessary.
    """
    frames = []
    for p in paths:
        frame = cv2.imread(p)
        if frame is not None:
            frames.append(frame)
    return frames
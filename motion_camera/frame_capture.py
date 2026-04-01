"""
frame_capture.py - Camera interface using picamera2.

Provides two things:
  - grab_frame()  : capture a single numpy frame (for idle motion checks)
  - record_visit(): capture frames at BURST_FPS until the caller signals
                    "stop" via a threading.Event, or MAX_VISIT_DURATION is hit.

All frames are returned as BGR numpy arrays (OpenCV convention) so the rest
of the pipeline needs no conversion.

This module does NOT print anything —-main.py handles all terminal output.
"""

import os
import time
import numpy as np
from picamera2 import Picamera2
import cv2

import config


def _make_camera() -> Picamera2:
    """
    Create and configure a Picamera2 instance.

    We use the 'still' configuration for idle frames (higher quality) and
    switch to a video-style configuration during burst capture (lower
    latency, consistent frame timing).
    """
    cam = Picamera2()
    # Start with a still config so the first grab_frame() call is clean.
    still_cfg = cam.create_still_configuration(
        main={"size": (config.CAPTURE_WIDTH, config.CAPTURE_HEIGHT),
              "format": "RGB888"},
        # Small lores stream for fast motion-check thumbnails
        lores={"size": (640, 480), "format": "YUV420"},
    )
    cam.configure(still_cfg)
    cam.start()
    # Allow the sensor to settle (AGC / AWB stabilisation)
    time.sleep(2.0)
    return cam


# Module-level singleton so we open the camera once and reuse it.
_camera: Picamera2 | None = None


def open_camera() -> None:
    """Open and warm up the camera. Call once at startup."""
    os.environ["LIBCAMERA_LOG_LEVELS"] = "3" # comment to enable picamera2 debug logs
    global _camera
    if _camera is None:
        _camera = _make_camera()


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
    Raises RuntimeError if the camera has not been opened.
    """
    if _camera is None:
        raise RuntimeError("Camera not open - call open_camera() first.")
    rgb = _camera.capture_array("main")
    # picamera2 RGB888 → OpenCV expects BGR
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def record_visit(stop_event) -> list[np.ndarray]:
    """
    Capture frames at BURST_FPS until stop_event is set or MAX_VISIT_DURATION
    elapses.  Returns the list of BGR frames at GIF resolution.

    Args:
        stop_event: threading.Event - caller sets this when the visit ends.

    Returns:
        List of BGR numpy arrays, each resized to (GIF_WIDTH, GIF_HEIGHT).
        May be empty if the camera fails immediately.

    Note on memory: 30 s * 8 fps = 240 frames * (1280*960*3) bytes ≈ 885 MB.
    That would OOM a 1 GB Pi.  We therefore cap storage at MAX_VISIT_DURATION
    and resize to GIF resolution (not full sensor resolution) during capture,
    keeping peak usage well under 200 MB for a 30-second burst.
    """
    if _camera is None:
        raise RuntimeError("Camera not open... call open_camera() first.")

    frames: list[np.ndarray] = []
    interval   = 1.0 / config.BURST_FPS
    start_time = time.monotonic()
    deadline   = start_time + config.MAX_VISIT_DURATION

    # Switch to video configuration for low-latency burst capture.
    # We reconfigure in-place so we don't lose the camera handle.
    _camera.stop()
    video_cfg = _camera.create_video_configuration(
        main={"size": (config.GIF_WIDTH, config.GIF_HEIGHT), "format": "RGB888"},
    )
    _camera.configure(video_cfg)
    _camera.start()

    try:
        while not stop_event.is_set() and time.monotonic() < deadline:
            loop_start = time.monotonic()
            rgb = _camera.capture_array("main")
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            frames.append(bgr)
            # Busy-wait for the remainder of the frame interval
            elapsed = time.monotonic() - loop_start
            remaining = interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
    finally:
        # Always restore still configuration so grab_frame() works again.
        _camera.stop()
        still_cfg = _camera.create_still_configuration(
            main={"size": (config.CAPTURE_WIDTH, config.CAPTURE_HEIGHT),
                  "format": "RGB888"},
            lores={"size": (640, 480), "format": "YUV420"},
        )
        _camera.configure(still_cfg)
        _camera.start()
        time.sleep(0.5)   # brief settle after reconfigure

    return frames

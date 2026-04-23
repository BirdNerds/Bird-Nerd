"""
frame_capture.py - Camera interface using picamera2.

Memory strategy
---------------
This program writes each frame to a temp JPEG on disk as it is captured,
then returns the list of file paths.  The caller (gif_builder.py) reads
them back one at a time, so peak RAM usage is roughly one frame at a time
rather than the entire burst.

White balance strategy
----------------------
The camera shoots through a window, often at dawn/dusk, which confuses AWB
into locking on a warm (orange/amber) cast.  The fix is a two-stage warmup:

  Stage 1 (5 s)  - camera runs freely; AWB converges on the scene
  Stage 2 (2 s)  - read back the ColourGains AWB settled on, then LOCK them

Locking the gains before burst capture means every frame in a visit uses the
same white balance, and the gains were chosen when the sensor was properly
warmed up rather than on the very first frame.

If ColourGains cannot be read (rare), we fall back to the Cloudy AWB preset,
which is neutral enough for most outdoor/window scenes and avoids the warm
Auto bias.

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
from libcamera import controls as lc
import config

os.environ["LIBCAMERA_LOG_LEVELS"] = "3"

_camera: Picamera2 | None = None
# Dedicated temp directory for burst frames - wiped on each new visit
_TEMP_DIR = "/tmp/bird_nerd_burst"


def _lock_white_balance(camera: Picamera2) -> None:
    """
    Read the ColourGains that AWB settled on, then freeze them.

    This prevents the orange/warm cast that appears when the camera is
    shooting through a window at dawn or dusk.  AWB in Auto mode can
    overcorrect for the warm ambient light and then lock that bad value
    for the rest of the session.

    Strategy:
      1. Give AWB 5 s to converge (called before this function).
      2. Read ColourGains from metadata.
      3. Disable AWB and write those gains back, locking colour for the session.
      4. If metadata read fails, fall back to Cloudy preset (neutral/cool bias).
    """
    try:
        metadata = camera.capture_metadata()
        colour_gains = metadata.get("ColourGains")

        if colour_gains and len(colour_gains) == 2:
            rg, bg = colour_gains

            # Sanity-check the gains.  Valid gains are typically 1.0–4.0.
            # An orange cast usually means rg >> bg (red boosted, blue crushed).
            # If red gain is more than 2× the blue gain, AWB overcorrected -
            # clamp red down and nudge blue up toward a neutral balance.
            if rg > 2.0 * bg:
                print(f"  AWB gains look warm (rg={rg:.2f}, bg={bg:.2f}) - clamping.")
                rg = min(rg, bg * 1.8)   # pull red back toward neutral

            print(f"  Locking white balance: red_gain={rg:.3f}, blue_gain={bg:.3f}")
            camera.set_controls({
                "AwbEnable":   False,
                "ColourGains": (rg, bg),
            })

        else:
            # Metadata read failed - fall back to a neutral preset.
            # Cloudy (mode 3) has a slightly cool bias that counteracts the
            # warm window cast better than Auto on problem scenes.
            print("  ColourGains not available - falling back to Cloudy AWB preset.")
            camera.set_controls({
                "AwbEnable": True,
                "AwbMode":   lc.AwbModeEnum.Cloudy,
            })

    except Exception as e:
        print(f"  White balance lock failed ({e}) - AWB left in Auto mode.")


def open_camera() -> None:
    """
    Open, configure, and warm up the camera.

    Two-stage warmup:
      Stage 1 (5 s): sensor and AWB converge freely.
      Stage 2 (2 s): ColourGains locked, exposure stabilises.

    Call once at startup.
    """
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

    # Stage 1: let AWB converge
    print("  Camera warming up (stage 1/2 - AWB converging)...")
    time.sleep(5.0)

    # Stage 2: lock the gains AWB settled on
    print("  Camera warming up (stage 2/2 - locking white balance)...")
    _lock_white_balance(_camera)
    time.sleep(2.0)

    print("  Camera ready.")


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
        raise RuntimeError("Camera not open - call open_camera() first.")
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

    White balance is NOT re-negotiated here - the gains locked in open_camera()
    stay in effect for the entire session.  Switching to video config and back
    preserves the locked ColourGains because we re-apply them after restoring
    the still config.

    Returns a list of JPEG file paths in capture order.
    Peak RAM during capture: ~1 frame = 1280*960*3 bytes ~ 3.5 MB.

    Args:
        stop_event: threading.Event - set by the caller when the bird leaves.

    Returns:
        List of absolute paths to JPEG files in _TEMP_DIR.
    """
    if _camera is None:
        raise RuntimeError("Camera not open - call open_camera() first.")

    # Remember the gains that are currently locked so we can re-apply them
    # after the video config switch (switching config resets some controls).
    try:
        metadata     = _camera.capture_metadata()
        locked_gains = metadata.get("ColourGains")
    except Exception:
        locked_gains = None

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

    # Re-apply locked gains after config switch
    if locked_gains and len(locked_gains) == 2:
        _camera.set_controls({
            "AwbEnable":   False,
            "ColourGains": locked_gains,
        })
    time.sleep(0.3)   # let the new config settle before capture

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

        # Re-apply locked gains on the restored still config too
        if locked_gains and len(locked_gains) == 2:
            _camera.set_controls({
                "AwbEnable":   False,
                "ColourGains": locked_gains,
            })
        time.sleep(0.5)

    return paths


def load_frames(paths: list[str]) -> list[np.ndarray]:
    """
    Read JPEG frames back from disk as BGR numpy arrays.
    Used by gif_builder and bird_classify when they need raw pixel data.
    Loads one at a time - caller should not hold the entire list in memory
    longer than necessary.
    """
    frames = []
    for p in paths:
        frame = cv2.imread(p)
        if frame is not None:
            frames.append(frame)
    return frames
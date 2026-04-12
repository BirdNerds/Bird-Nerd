"""
main.py - Bird Nerd main loop and terminal output.

This file owns:
  - All terminal print() output
  - Accurate wall-clock timing (fixed in this version)
  - The top-level visit state machine
  - Ctrl+C shutdown

Visit state machine
-------------------
IDLE        waiting for motion inside the ROI
RECORDING   bird detected; burst capture running in a background thread;
            motion checked each frame to detect departure
PROCESSING  burst finished; classify, build GIF, upload, log
-------------------------------------------------
"""

import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import config
import firebase_upload
import frame_capture
import bird_classify
import gif_builder
import motion_detect
import sighting_log


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------

def _ensure_dirs() -> None:
    for d in (config.IMAGES_DIR, config.UNCLEAR_DIR, config.GIFS_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)


def _check_model() -> bool:
    return os.path.exists(config.MODEL_PATH) and os.path.exists(config.LABELS_PATH)


def _print_header(firebase_ok: bool, model_ok: bool) -> None:
    print("=" * 60)
    print("Bird Nerd - AI window-feeder classifier")
    print("=" * 60)
    print(f"  Log file  : {config.LOG_FILE}")
    print(f"  Images    : {config.IMAGES_DIR}")
    print(f"  GIFs      : {config.GIFS_DIR}")
    print(f"  Model     : {'loaded' if model_ok else 'NOT FOUND - motion-only mode'}")
    print(f"  Firebase  : {'connected' if firebase_ok else 'disabled'}")
    print(f"  ROI       : top={config.ROI_TOP:.0%}  bottom={config.ROI_BOTTOM:.0%}"
          f"  left={config.ROI_LEFT:.0%}  right={config.ROI_RIGHT:.0%}")
    print(f"  Burst     : {config.BURST_FPS} fps  max {config.MAX_VISIT_DURATION:.0f}s"
          f"  classify every {config.CLASSIFY_EVERY_N_FRAMES} frames")
    print("=" * 60)
    print("Press Ctrl+C to stop.\n")


# ---------------------------------------------------------------------------
# Visit recording (runs in a background thread)
# ---------------------------------------------------------------------------

def _record_visit_thread(stop_event: threading.Event, result_box: list) -> None:
    """
    Capture burst frames until stop_event is set or the hard deadline.
    Stores the frame list in result_box[0].
    """
    frames = frame_capture.record_visit(stop_event)
    result_box.append(frames)


# ---------------------------------------------------------------------------
# Post-visit processing
# ---------------------------------------------------------------------------

def _process_visit(
    frame_paths: list,
    classifier: "bird_classify.BirdClassifier | None",
    firebase_ok: bool,
    visit_start: datetime,
) -> None:
    """
    Classify the burst, build a GIF, save a thumbnail, log and upload.
    Accepts file paths rather than numpy arrays - frames stay on disk
    until needed, keeping RAM flat.
    All print() statements for the outcome live here.
    """
    now_str = datetime.now().strftime("%H:%M:%S")

    if not frame_paths:
        print(f"{now_str}  Visit ended with no frames captured.")
        return

    print(f"{now_str}  Visit ended - {len(frame_paths)} frames captured.")

    # -- Classification (vote() loads frames from disk one at a time) -----
    result = None
    if classifier is not None:
        result = classifier.vote(frame_paths)

    if result is None:
        label      = "unknown"
        confidence = 0.0
        top_3:    list = []
        thumb_dir  = config.UNCLEAR_DIR
        print(f"{now_str}  Classification: inconclusive (not enough texture).")
    else:
        label      = result.label
        confidence = result.confidence
        top_3      = result.top_3
        thumb_dir  = (config.IMAGES_DIR
                      if confidence >= config.CONFIDENCE_THRESHOLD
                      else config.UNCLEAR_DIR)
        status = "Detected" if confidence >= config.CONFIDENCE_THRESHOLD else "Low confidence"
        print(f"{now_str}  {status}: {label} ({confidence:.2%})"
              f"  [{result.vote_count}/{result.frame_count} frames agreed]")

    # -- Build GIF + thumbnail --------------------------------------------
    try:
        gif_path, thumb_path = gif_builder.build(
            frame_paths = frame_paths,
            label       = label,
            thumb_dir   = thumb_dir,
            timestamp   = visit_start,
        )
        print(f"{now_str}  Saved GIF      : {os.path.basename(gif_path)}")
        print(f"{now_str}  Saved thumbnail: {os.path.basename(thumb_path)}")
    except Exception as e:
        print(f"{now_str}  GIF/thumbnail save failed: {e}")
        gif_path   = None
        thumb_path = None

    # -- Text log ----------------------------------------------------------
    if result is not None:
        sighting_log.log_sighting(label, confidence, top_3, timestamp=visit_start)

    # -- Firebase upload ---------------------------------------------------
    if firebase_ok and result is not None:
        sighting = {
            "common_name":       label,
            "scientific_name":   label,  # full label string includes both
            "confidence":        confidence,
            "top_3_predictions": top_3,
            "thumb_path":        thumb_path,
            "gif_path":          gif_path,
            "timestamp":         visit_start,
            "timezone":          str(config.LOCAL_TIMEZONE),
        }
        doc_id = firebase_upload.upload_sighting(sighting)
        if doc_id:
            print(f"{now_str}  Firebase doc   : {doc_id[:12]}...")
        else:
            print(f"{now_str}  Firebase upload failed.")

    print()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    program_start = time.monotonic()   # wall-clock reference for accurate runtime
    check_count   = 0
    visit_count   = 0

    _ensure_dirs()

    # -- Firebase ----------------------------------------------------------
    firebase_ok = firebase_upload.initialize()

    # -- Classifier --------------------------------------------------------
    model_ok = _check_model()
    classifier: bird_classify.BirdClassifier | None = None
    if model_ok:
        try:
            classifier = bird_classify.BirdClassifier()
        except Exception as e:
            print(f"Model load failed: {e} - running in motion-only mode.")
            model_ok = False

    _print_header(firebase_ok, model_ok)

    # -- Camera ------------------------------------------------------------
    frame_capture.open_camera()
    print("Camera warming up...")
    time.sleep(5.0)

    # Capture initial reference frame
    prev_roi = motion_detect.apply_roi(frame_capture.grab_frame())
    print("Ready - monitoring for birds.\n")
    if firebase_ok:
        firebase_upload.send_heartbeat(alive=True)

    try:
        while True:
            time.sleep(config.IDLE_CHECK_INTERVAL)

            current_frame = frame_capture.grab_frame()
            current_roi   = motion_detect.apply_roi(current_frame)
            check_count  += 1

            motion, area, bboxes = motion_detect.detect_motion(prev_roi, current_roi)

            if not motion:
                prev_roi = current_roi
                # Print a heartbeat every 60 checks (~24 s at 0.4 s interval)
                if check_count % 60 == 0:
                    elapsed_min = (time.monotonic() - program_start) / 60
                    print(f"  Monitoring... "
                          f"{elapsed_min:.1f} min elapsed, "
                          f"{visit_count} visit(s) recorded.")
                    if firebase_ok:
                        firebase_upload.send_heartbeat(alive=True)

                continue

            # ---- Motion detected - start a visit -------------------------
            visit_start = datetime.now(config.LOCAL_TIMEZONE)
            now_str     = visit_start.strftime("%H:%M:%S")
            print(f"{now_str}  Motion detected (area={area:.0f} px).  Recording visit...")
            visit_count += 1

            stop_event  = threading.Event()
            result_box: list = []

            # Start burst capture in background thread
            recorder_thread = threading.Thread(
                target=_record_visit_thread,
                args=(stop_event, result_box),
                daemon=True,
            )
            recorder_thread.start()

            # Monitor motion in the foreground to decide when the bird leaves.
            # We stop the recording when there is no motion for NO_MOTION_TIMEOUT
            # seconds, or when MAX_VISIT_DURATION elapses (the record_visit()
            # function also enforces the hard deadline independently).
            last_motion_time = time.monotonic()
            visit_deadline   = last_motion_time + config.MAX_VISIT_DURATION

            # During burst capture the camera is in video (GIF) resolution mode.
            # grab_frame() would return a different resolution than current_roi,
            # causing the OpenCV size-mismatch crash.  Instead we monitor the
            # temp directory: as long as new frames are being written to disk the
            # bird is still being recorded.  We declare the visit over when no new
            # frame has appeared for NO_MOTION_TIMEOUT seconds, or the deadline hits.
            last_frame_count = 0
            last_new_frame_time = time.monotonic()
            while time.monotonic() < visit_deadline:
                time.sleep(config.IDLE_CHECK_INTERVAL)
                current_frame_count = len([
                    f for f in os.listdir("/tmp/bird_nerd_burst")
                    if f.endswith(".jpg")
                ])
                if current_frame_count > last_frame_count:
                    last_frame_count    = current_frame_count
                    last_new_frame_time = time.monotonic()
                elif time.monotonic() - last_new_frame_time >= config.NO_MOTION_TIMEOUT:
                    break   # no new frames written - bird likely gone

            stop_event.set()
            recorder_thread.join(timeout=5.0)

            frame_paths = result_box[0] if result_box else []
            _process_visit(frame_paths, classifier, firebase_ok, visit_start)

            # Update reference frame after the visit so a stationary background
            # doesn't immediately re-trigger.
            prev_roi = motion_detect.apply_roi(frame_capture.grab_frame())

    except KeyboardInterrupt:
        elapsed_sec = time.monotonic() - program_start
        elapsed_min = elapsed_sec / 60

        print("\n" + "=" * 60)
        print("Bird Nerd stopped.")
        print(f"  Runtime    : {elapsed_min:.1f} min ({elapsed_sec:.0f} s)")
        print(f"  Checks     : {check_count}")
        print(f"  Visits     : {visit_count}")
        print("=" * 60)
        if firebase_ok:
            firebase_upload.send_heartbeat(alive = False)

    except Exception as e:
        import traceback
        print(f"\nFatal error: {e}")
        traceback.print_exc()
        sys.exit(1)

    finally:
        frame_capture.close_camera()


if __name__ == "__main__":
    main()
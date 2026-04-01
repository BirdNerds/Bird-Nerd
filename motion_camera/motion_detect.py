"""
motion_detect.py - Region-of-interest extraction and motion detection.

Two public functions:
  apply_roi(frame)          -> cropped numpy array
  detect_motion(f1, f2)     -> (bool, total_area_px, list_of_bboxes)

Both operate on BGR numpy arrays.

ROI rationale
-------------
The camera is mounted inside at a second-floor window looking at a tray
feeder ~30 cm away.  The full 2592*1944 frame includes:
  - Upper ~30 %  : sky and treetops  → lots of lighting-change false triggers
  - Lower ~15 %  : bottom edge of window frame / feeder box interior
  - Side ~10 %   : window frame on each side

Cropping to ROI_TOP/BOTTOM/LEFT/RIGHT (defined in config.py) reduces the
classification input to the zone where birds actually appear, dramatically
cutting false positives from background movement.
"""

import cv2
import numpy as np

import config


# ---------------------------------------------------------------------------
# ROI
# ---------------------------------------------------------------------------

def apply_roi(frame: np.ndarray) -> np.ndarray:
    """
    Return the region-of-interest sub-image from a full frame.

    The ROI is defined as fractional coordinates in config.py so it stays
    correct regardless of capture resolution.

    Args:
        frame: BGR numpy array of any resolution.

    Returns:
        BGR numpy array - the cropped ROI.  Never returns an empty array
        as long as the config fractions are sane (checked at startup).
    """
    h, w = frame.shape[:2]
    y1 = int(h * config.ROI_TOP)
    y2 = int(h * config.ROI_BOTTOM)
    x1 = int(w * config.ROI_LEFT)
    x2 = int(w * config.ROI_RIGHT)
    return frame[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Motion detection
# ---------------------------------------------------------------------------

def detect_motion(
    frame1: np.ndarray,
    frame2: np.ndarray,
) -> tuple[bool, float, list[tuple[int, int, int, int]]]:
    """
    Compare two consecutive frames (already ROI-cropped) and return whether
    meaningful motion occurred inside that region.

    Algorithm:
      1. Convert both frames to greyscale and apply Gaussian blur
         (removes high-frequency noise and compression artefacts).
      2. Compute absolute pixel difference.
      3. Threshold and dilate to merge nearby change regions into blobs.
      4. Find contours; keep only those larger than MIN_CONTOUR_AREA.
      5. If the largest blob covers > LARGE_MOTION_RATIO of the ROI it is
         almost certainly a global lighting change (cloud passing, car
         headlights) - return no motion.

    Args:
        frame1: BGR numpy array - the earlier frame (ROI-cropped).
        frame2: BGR numpy array - the later frame (ROI-cropped).

    Returns:
        (motion_detected, total_area_px, bboxes)
        - motion_detected : True if at least one qualifying contour found.
        - total_area_px   : sum of qualifying contour areas.
        - bboxes          : list of (x, y, w, h) for each qualifying contour,
                            coordinates relative to the ROI origin.
    """
    # -- Greyscale + blur --------------------------------------------------
    blur = config.MOTION_BLUR_SIZE
    g1 = cv2.GaussianBlur(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), (blur, blur), 0)
    g2 = cv2.GaussianBlur(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), (blur, blur), 0)

    # -- Difference + threshold + dilate -----------------------------------
    diff   = cv2.absdiff(g1, g2)
    _, thr = cv2.threshold(diff, config.MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
    thr    = cv2.dilate(thr, None, iterations=2)

    # -- Contours ----------------------------------------------------------
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roi_area = frame1.shape[0] * frame1.shape[1]  # height * width of cropped ROI

    total_area  = 0.0
    bboxes: list[tuple[int, int, int, int]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < config.MIN_CONTOUR_AREA:
            continue
        bboxes.append(cv2.boundingRect(cnt))
        total_area += area

    if not bboxes:
        return False, 0.0, []

    # -- Large-motion guard -----------------------------------------------
    # If the single biggest blob dominates the ROI it's probably not a bird.
    largest_area = max(cv2.contourArea(c) for c in contours)
    if largest_area / roi_area >= config.LARGE_MOTION_RATIO:
        return False, 0.0, []

    return True, total_area, bboxes

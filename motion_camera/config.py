"""
config.py - Bird Nerd configuration
All tunable constants live here. Nothing else imports from anywhere
except this file for settings.
"""

import os
import pytz

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR   = os.path.join(SCRIPT_DIR, "images")          # high-confidence stills
UNCLEAR_DIR  = os.path.join(SCRIPT_DIR, "unclear_images")  # low-confidence stills
GIFS_DIR     = os.path.join(SCRIPT_DIR, "gifs")            # all saved GIFs
LOG_FILE     = os.path.join(SCRIPT_DIR, "sightings.log")

MODEL_PATH   = os.path.join(SCRIPT_DIR, "models", "bird_classifier.tflite")
LABELS_PATH  = os.path.join(SCRIPT_DIR, "models", "labels.txt")

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
# Full sensor resolution for the X000VGJ8BL camera module
CAPTURE_WIDTH  = 2592
CAPTURE_HEIGHT = 1944

# GIF output resolution (, keeps file sizes reasonable on 1 GB Pi)
GIF_WIDTH  = 640
GIF_HEIGHT = 480

# Frames per second during a bird visit burst capture
# picamera2 can comfortably do ~8 fps at GIF resolution on a Pi 4
BURST_FPS = 6

# ---------------------------------------------------------------------------
# Region of Interest (ROI)
# Applied before motion detection AND before classification.
#
# All values are fractions of the full frame (0.0 - 1.0).
# ---------------------------------------------------------------------------
ROI_TOP    = 0.25   # crop from 25 % down (removes most sky, catches high perches)
ROI_BOTTOM = 0.92   # crop to  92 % (includes full feeder tray lip)
ROI_LEFT   = 0.05   # crop from  5 % in (catches birds on left feeder edge)
ROI_RIGHT  = 0.95   # crop to  95 % in (catches birds on right feeder edge)

# ---------------------------------------------------------------------------
# Motion detection
# ---------------------------------------------------------------------------
MOTION_BLUR_SIZE  = 21     # Gaussian blur kernel (must be odd)
MOTION_THRESHOLD  = 75     # pixel-difference threshold after blur
MIN_CONTOUR_AREA  = 12000  # px^2 - contours smaller than this are ignored
                            # (eliminates single-leaf flutter, insects)

# If the largest motion contour covers more than this fraction of the ROI,
# it is almost certainly a lighting change or window glare - skip it.
LARGE_MOTION_RATIO = 0.35

# ---------------------------------------------------------------------------
# Visit session timing
# ---------------------------------------------------------------------------
# How long to wait with no motion before deciding that the bird has left.
NO_MOTION_TIMEOUT = 4.0    # seconds

# Hard cap on a single visit's GIF length regardless of motion.
# IRL window-feeder visits are typically 5 to 25 seconds
MAX_VISIT_DURATION = 10.0  # seconds

# Brief pause between motion-check frames when no visit is active.
IDLE_CHECK_INTERVAL = 0.4  # seconds

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
# Minimum confidence to log a sighting to images/ and Firebase.
# Below this threshold the still goes to unclear_images/ and is NOT uploaded.
CONFIDENCE_THRESHOLD = 0.60

# Run inference on every Nth frame during a burst to stay lightweight.
# At 8 fps, classifying every 4th frame = ~2 inferences/sec, well within
# Pi 4 limits (~200-500 ms per inference on a cropped ROI).
CLASSIFY_EVERY_N_FRAMES = 4

# Minimum number of frame votes needed before declaring a winner.
# Prevents a single fluke frame from dictating the species.
MIN_VOTES_REQUIRED = 2

# Pre-classification texture check: if the Laplacian variance of the crop
# is below this value the image is too smooth to contain a bird - skip.
LAP_VAR_THRESHOLD = 18.0

# ---------------------------------------------------------------------------
# Image preprocessing - CLAHE
# Applied to every frame before TFLite inference to improve accuracy under
# tricky lighting (colour casts, backlit birds, sunrise/overcast glare).
#
# CLAHE (Contrast Limited Adaptive Histogram Equalization) operates on the
# luminance channel in LAB colour space so it boosts local contrast without
# distorting hue or saturation.
#
# CLAHE_CLIP_LIMIT   - controls how aggressively contrast is boosted.
#                      2.0  = conservative, safe default
#                      3.0  = stronger correction, good for colour-cast frames
#                      4.0  = aggressive; use if 3.0 still looks washed out
#
# CLAHE_TILE_SIZE    - grid size for local equalization (width, height).
#                      (8, 8) is the standard starting point. Smaller tiles
#                      = more localised correction but can introduce artifacts.
# ---------------------------------------------------------------------------
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_SIZE  = (8, 8)

# ---------------------------------------------------------------------------
# Timezone
# ---------------------------------------------------------------------------
LOCAL_TIMEZONE = pytz.timezone("America/New_York")  # adjust if you move

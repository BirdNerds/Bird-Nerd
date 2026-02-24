"""
Bird Detection and Classification using TensorFlow Lite
Combines motion detection with on-device ML classification
Based on: https://mikesml.com/2021/05/16/image-recognition-on-the-edge-tflite-on-raspberry-pi/
"""

import cv2
import numpy as np
import subprocess
import time
import os
from datetime import datetime
import pytz
from pathlib import Path
import warnings

try:
    import firebase_helper
    FIREBASE_ENABLED = True
    firebase_helper.initialize_firebase()
except ImportError:
    print("firebase_helper not found - Firebase logging disabled")
    FIREBASE_ENABLED = False
except Exception as e:
    print(f"Firebase initialization failed: {e}")
    FIREBASE_ENABLED = False

# Suppress noisy NumPy getlimits UserWarning about smallest subnormal being zero
warnings.filterwarnings(
    "ignore",
    message="The value of the smallest subnormal for <class 'numpy.float64'> type is zero.",
    category=UserWarning,
    module="numpy.core.getlimits"
)

# TensorFlow Lite imports
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    print("tflite_runtime not found, trying tensorflow")
    import tensorflow as tf

# ============= CONFIGURATION =============
# Directories - ALL relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")
UNCLEAR_DIR = os.path.join(SCRIPT_DIR, "unclear_images")
TEMP_DIR = "/tmp/bird_feeder" # Used for temporary captures; not critical to keep long-term
LOG_FILE = os.path.join(SCRIPT_DIR, "sightings.log")

# Model Settings
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "bird_classifier.tflite")
LABELS_PATH = os.path.join(SCRIPT_DIR, "models", "labels.txt")
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to save to /images/ any lower goes to /unclear/

# Motion Detection Settings
MOTION_THRESHOLD = 40
MIN_AREA = 2000
BLUR_SIZE = 21

# Large-motion and bbox padding heuristics
LARGE_MOVE_RATIO = 0.6   # If largest bbox covers >60% of frame, skip classification
BBOX_PADDING = 1.25      # Scale bbox by this factor when cropping for classification

# Pre-classification heuristics to avoid classifying smooth backgrounds
LAP_VAR_THRESHOLD = 12.0         # variance of Laplacian below this => too smooth (skip)
ASPECT_RATIO_RANGE = (0.08, 12.0) # acceptable w/h ratio for candidate crop

# Camera Settings
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
CAPTURE_TIMEOUT = 2000 # milliseconds

# Timing
CHECK_INTERVAL = 0.5 # seconds between motion checks
COOLDOWN_PERIOD = 10.0 # seconds to wait after a detection
# Real world: COOLDOWN_PERIOD = 30 seconds
# Testing: COOLDOWN_PERIOD = 10 seconds

# Timezone for logging/database
LOCAL_TIMEZONE = pytz.timezone('America/New_York')  # EST/EDT

# ============= SETUP =============
def setup_directories():
    """Create necessary directories"""
    for directory in [IMAGES_DIR, UNCLEAR_DIR, TEMP_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print("Directories created")

def load_labels(path):
    """Load labels from text file"""
    try:
        with open(path, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        return labels
    except FileNotFoundError:
        print(f"Warning: Labels file not found at {path}")
        return []

class BirdClassifier:
    """TensorFlow Lite bird classifier"""
    
    def __init__(self, model_path, labels_path):
        self.labels = load_labels(labels_path)
        
        # Load TFLite model
        try:
            self.interpreter = tflite.Interpreter(model_path=model_path)
        except Exception as e:
            # Fallback to tensorflow if tflite_runtime not available
            print(f"   tflite_runtime failed: {e}")
            print(f"   Trying tensorflow.lite...")
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
        
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get expected input shape
        self.input_shape = self.input_details[0]['shape']
        self.height = self.input_shape[1]
        self.width = self.input_shape[2]
        
        print(f"Model loaded: {model_path}")
        print(f"  Input shape: {self.input_shape}")
        print(f"  Labels loaded: {len(self.labels)}")
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # Resize to model's expected input size
        resized = cv2.resize(image, (self.width, self.height))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] if model expects float input
        if self.input_details[0]['dtype'] == np.float32:
            rgb = rgb.astype(np.float32) / 255.0
        
        # Add batch dimension
        input_data = np.expand_dims(rgb, axis=0)
        
        return input_data
    
    def classify(self, image):
        """
        Classify the image
        Returns: (label, confidence, all_predictions)
        """
        # Preprocess
        input_data = self.preprocess_image(image)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get results
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        predictions = output_data[0]
        
        # Robust handling: ensure float, replace non-finite, clip to safe range
        predictions = predictions.astype(np.float64)
        # Replace NaN/inf with large negative so they won't dominate softmax
        predictions = np.nan_to_num(predictions, nan=-1e9, posinf=1e9, neginf=-1e9)
        # Clip to avoid overflow in exp
        predictions = np.clip(predictions, -100.0, 100.0)
        
        # Stable softmax
        shifted = predictions - np.max(predictions)
        exp_predictions = np.exp(shifted)
        sum_exp = np.sum(exp_predictions)
        if sum_exp == 0 or not np.isfinite(sum_exp):
            # Fallback: no confident output, produce zeros
            probabilities = np.zeros_like(exp_predictions)
        else:
            probabilities = exp_predictions / sum_exp
        
        # If probabilities contain non-finite values, zero them out
        if not np.all(np.isfinite(probabilities)):
            probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Get top prediction
        top_idx = int(np.argmax(probabilities)) if probabilities.size else 0
        confidence = float(probabilities[top_idx]) if probabilities.size else 0.0
        
        # Get label
        if top_idx < len(self.labels):
            label = self.labels[top_idx]
        else:
            label = f"class_{top_idx}"
        
        # Get top 3 predictions
        top_3_idx = np.argsort(probabilities)[-3:][::-1] if probabilities.size else []
        top_3 = [(self.labels[i] if i < len(self.labels) else f"class_{i}", 
                  float(probabilities[i]) if probabilities.size else 0.0) for i in top_3_idx]
        
        return label, confidence, top_3

def capture_image(filename):
    """Capture image using rpicam-still"""
    try:
        result = subprocess.run(
            ['rpicam-still', '-o', filename, '-t', str(CAPTURE_TIMEOUT),
             '--width', str(IMAGE_WIDTH), '--height', str(IMAGE_HEIGHT),
             '-n'],
            capture_output=True, 
            text=True,
            timeout=5
        )
        return result.returncode == 0 and os.path.exists(filename)
    except Exception as e:
        print(f"Error capturing: {e}")
        return False

def detect_motion(frame1, frame2):
    """Compare frames and detect motion. Returns (motion_detected, total_area, bboxes).

    bboxes: list of (x, y, w, h) for contours whose area > MIN_AREA
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    gray1 = cv2.GaussianBlur(gray1, (BLUR_SIZE, BLUR_SIZE), 0)
    gray2 = cv2.GaussianBlur(gray2, (BLUR_SIZE, BLUR_SIZE), 0)
    
    frame_delta = cv2.absdiff(gray1, gray2)
    thresh = cv2.threshold(frame_delta, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    motion_detected = False
    total_area = 0
    bboxes = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > MIN_AREA:
            motion_detected = True
            total_area += area
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append((x, y, w, h))
    
    return motion_detected, total_area, bboxes

def log_detection(label, scientific_name, confidence, top_3):
    """
    Append detection to bird_sightings.log in the specified format
    """
    with open(LOG_FILE, 'a') as f:
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp}\n")
        
        # Common name
        f.write(f"{label}\n")
        
        # Scientific name with confidence
        f.write(f"{scientific_name} ({confidence:.2%})\n")
        
        # Blank line
        f.write("\n")
        
        # Top 3 predictions
        f.write("Top 3 predictions:\n")
        for i, (lbl, conf) in enumerate(top_3, 1):
            # Extract scientific name from label (assumes format like in labels.txt)
            # If label contains both common and scientific, parse it
            # For now, we'll just use the label as-is
            f.write(f"{i}. {lbl}: {conf:.2%}\n")
        
        # Footer separator
        f.write("\n**********************************************************************\n")

def save_classified_image(frame, label, confidence, top_3):
    """
    Save image to /images/ or /unclear/ based on confidence
    Filename format: year_month_day_hour_minute_second_commonname.jpg
    Log to bird_sightings.log AND Firebase
    """
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    time_str = datetime.now().strftime("%H:%M:%S")
    
    # Clean label for filename (replace spaces with underscores, remove special chars)
    clean_label = label.replace(' ', '_').replace('(', '').replace(')', '')
    clean_label = ''.join(c for c in clean_label if c.isalnum() or c == '_')
    
    # Determine save location based on confidence
    if confidence >= CONFIDENCE_THRESHOLD:
        save_dir = IMAGES_DIR
        status = "Detected"
    else:
        save_dir = UNCLEAR_DIR
        status = "Low confidence"
    
    # Create filename
    filename = f"{timestamp}_{clean_label}.jpg"
    filepath = os.path.join(save_dir, filename)
    
    # Save image
    cv2.imwrite(filepath, frame)
    
    print(f"{time_str}  {status}: {label} ({confidence:.2%}) -> {filename}")
    
    # Log to bird_sightings.log
    scientific_name = label  # Modify if your labels have a different format
    log_detection(label, scientific_name, confidence, top_3)
    
    # Log to Firebase if enabled (pass filepath so the image is uploaded too)
    if FIREBASE_ENABLED:
        try:
            firebase_helper.add_bird_sighting(
                common_name=label,
                scientific_name=scientific_name,
                confidence=confidence,
                top_3_predictions=top_3,
                image_path=filepath
            )
        except Exception as e:
            print(f"Warning: Failed to log to Firebase: {e}")
    
    return filepath

def main():
    """Main detection loop"""
    print("Welcome to Bird Nerd, the AI bird classifier for your backyard!")
    
    setup_directories()
    
    print(f"Images directory: {IMAGES_DIR}")
    # print(f"Unclear directory: {UNCLEAR_DIR}")
    print(f"Log file: {LOG_FILE}")
    print(f"Pre-classify thresholds: lap_var >= {LAP_VAR_THRESHOLD}, aspect_ratio in {ASPECT_RATIO_RANGE}")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\ERROR: Model not found at {MODEL_PATH}")
        print("\nTo get started, you need a TFLite model")
        print("Bird Nerd devs has used this Google AIY model in the past")
        print("https://aiyprojects.withgoogle.com/model/nature-explorer/")
        print("\nFor now, running in motion-only mode...\n")
        classifier = None
    else:
        try:
            classifier = BirdClassifier(MODEL_PATH, LABELS_PATH)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Running in motion-only mode...\n")
            classifier = None
    
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD} any lower goes to {UNCLEAR_DIR}")
    print("\nStarting... Press Ctrl+C to stop\n")
    
    # Temporary image paths
    temp_img1 = os.path.join(TEMP_DIR, "temp1.jpg")
    temp_img2 = os.path.join(TEMP_DIR, "temp2.jpg")
    
    # Capture initial frame
    print("Capturing initial frame...")
    if not capture_image(temp_img1):
        print("Failed to capture initial frame")
        return
    
    prev_frame = cv2.imread(temp_img1)
    if prev_frame is None:
        print("Failed to read initial frame")
        return
    
    print("Ready! Monitoring for birds...\n")
    
    detection_count = 0
    check_count = 0
    
    try:
        while True:
            # Capture new frame
            if not capture_image(temp_img2):
                time.sleep(CHECK_INTERVAL)
                continue
            
            current_frame = cv2.imread(temp_img2)
            if current_frame is None:
                time.sleep(CHECK_INTERVAL)
                continue
            
            # Detect motion (now returns bboxes)
            motion_detected, area, bboxes = detect_motion(prev_frame, current_frame)
            check_count += 1
            
            if motion_detected:
                now_str = datetime.now().strftime("%H:%M:%S")
                # print(f"{now_str}  Motion detected! Area: {area}px")
                print(f"{now_str}  Motion detected! Area: {area}px")
                
                # Classify if model available
                if classifier:
                    # Choose largest bbox (if any) to crop the region of interest
                    if bboxes:
                        # pick bbox with largest area
                        x, y, w, h = max(bboxes, key=lambda b: b[2] * b[3])
                        bbox_area = w * h
                        frame_area = IMAGE_WIDTH * IMAGE_HEIGHT
                        bbox_ratio = bbox_area / float(frame_area)
                        
                        # If bbox is huge relative to frame, it's probably background/camera motion -> skip classify
                        if bbox_ratio >= LARGE_MOVE_RATIO:
                            print(f"{now_str}  Skipping classification (large motion region: {bbox_ratio:.2%})")
                            detection_count += 1
                        else:
                            # Expand bbox with padding and clip to frame bounds
                            cx = x + w // 2
                            cy = y + h // 2
                            new_w = int(w * BBOX_PADDING)
                            new_h = int(h * BBOX_PADDING)
                            x1 = max(0, cx - new_w // 2)
                            y1 = max(0, cy - new_h // 2)
                            x2 = min(IMAGE_WIDTH, cx + new_w // 2)
                            y2 = min(IMAGE_HEIGHT, cy + new_h // 2)
                            
                            # Crop and classify the ROI
                            crop = current_frame[y1:y2, x1:x2]
                            
                            # Fallback if crop empty for any reason
                            if crop.size == 0:
                                print(f"{now_str}  Warning: empty crop; skipping classification")
                                detection_count += 1
                            else:
                                # Pre-classification checks: aspect ratio and Laplacian variance
                                ch, cw = crop.shape[:2]
                                ar = (cw / float(ch)) if ch > 0 else 0.0
                                if not (ASPECT_RATIO_RANGE[0] <= ar <= ASPECT_RATIO_RANGE[1]):
                                    print(f"{now_str}  Skipping classification (bad aspect ratio: {ar:.2f})")
                                    detection_count += 1
                                else:
                                    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                                    lap_var = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
                                    if lap_var < LAP_VAR_THRESHOLD:
                                        print(f"{now_str}  Skipping classification (low texture: lap_var={lap_var:.2f})")
                                        detection_count += 1
                                    else:
                                        # CLASSIFY!
                                        label, confidence, top_3 = classifier.classify(crop)
                                        
                                        # Save image and log detection
                                        save_classified_image(current_frame, label, confidence, top_3)
                                        
                                        detection_count += 1
                    else:
                        # No bbox (shouldn't happen if motion_detected)
                        print(f"{now_str}  Motion detected but no bbox found")
                        detection_count += 1
                else:
                    # No classifier, just count motion
                    detection_count += 1
                
                time.sleep(COOLDOWN_PERIOD)
            else:
                if check_count % 10 == 0:
                    print(f"Monitoring... (checks: {check_count}, detections: {detection_count})")
            
            prev_frame = current_frame.copy()
            time.sleep(CHECK_INTERVAL)
    
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Detection stopped")
        print(f"Total checks: {check_count}")
        print(f"Detections: {detection_count}")
        print(f"Program ran for {(check_count * CHECK_INTERVAL) / 60:.1f} minutes")
        print("=" * 60)
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
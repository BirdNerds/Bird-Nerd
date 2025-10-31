#!/usr/bin/env python3
"""
Motion Detection Script for Raspberry Pi Bird Feeder
Uses rpicam-still for image capture and OpenCV for motion detection
Compatible with Raspberry Pi 4 and ArduCam modules using libcamera
"""

import cv2
import numpy as np
import subprocess
import time
import os
from datetime import datetime
from pathlib import Path

# ============= CONFIGURATION =============
# Directories
MOTION_DIR = "/home/tocila/Documents/pi-timolo/media/motion"
TEMP_DIR = "/tmp/bird_feeder"
ARCHIVE_DIR = "/home/tocila/Documents/pi-timolo/media/archive"

# Motion Detection Settings
MOTION_THRESHOLD = 25  # Lower = more sensitive (adjust between 15-40)
MIN_AREA = 500  # Minimum area in pixels to trigger motion (adjust 300-1000)
BLUR_SIZE = 21  # Gaussian blur kernel size (must be odd number)

# Camera Settings
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
CAPTURE_TIMEOUT = 2000  # milliseconds

# Timing
CHECK_INTERVAL = 0.5  # seconds between motion checks
COOLDOWN_PERIOD = 2.0  # seconds to wait after detecting motion before checking again

# Image Settings
IMAGE_PREFIX = "mo-"
SAVE_COMPARISON_IMAGES = False  # Set to True for debugging motion detection

# ============= SETUP =============
def setup_directories():
    """Create necessary directories if they don't exist"""
    for directory in [MOTION_DIR, TEMP_DIR, ARCHIVE_DIR]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print(f"✓ Directories created/verified")

def capture_image(filename):
    """Capture image using rpicam-still"""
    try:
        result = subprocess.run(
            ['rpicam-still', '-o', filename, '-t', str(CAPTURE_TIMEOUT),
             '--width', str(IMAGE_WIDTH), '--height', str(IMAGE_HEIGHT),
             '-n'],  # -n disables preview window
            capture_output=True, 
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            print(f"Camera capture failed: {result.stderr}")
            return False
        
        return os.path.exists(filename)
    
    except subprocess.TimeoutExpired:
        print("Camera capture timed out")
        return False
    except Exception as e:
        print(f"Error capturing image: {e}")
        return False

def detect_motion(frame1, frame2):
    """
    Compare two frames and detect motion
    Returns: (motion_detected, motion_area, diff_image)
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Blur to reduce noise
    gray1 = cv2.GaussianBlur(gray1, (BLUR_SIZE, BLUR_SIZE), 0)
    gray2 = cv2.GaussianBlur(gray2, (BLUR_SIZE, BLUR_SIZE), 0)
    
    # Compute difference
    frame_delta = cv2.absdiff(gray1, gray2)
    thresh = cv2.threshold(frame_delta, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    
    # Dilate to fill in holes
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if any contour is large enough
    motion_detected = False
    total_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > MIN_AREA:
            motion_detected = True
            total_area += area
    
    return motion_detected, total_area, thresh

def save_motion_image(frame, area):
    """Save image with timestamp when motion is detected"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{IMAGE_PREFIX}{timestamp}.jpg"
    filepath = os.path.join(MOTION_DIR, filename)
    
    cv2.imwrite(filepath, frame)
    print(f"🐦 Motion detected! Area: {area}px | Saved: {filename}")
    
    return filepath

def main():
    """Main motion detection loop"""
    print("=" * 60)
    print("Bird Feeder Motion Detection")
    print("=" * 60)
    print(f"Motion Directory: {MOTION_DIR}")
    print(f"Motion Threshold: {MOTION_THRESHOLD}")
    print(f"Minimum Area: {MIN_AREA}px")
    print(f"Image Size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print("=" * 60)
    print("Starting motion detection... Press Ctrl+C to stop")
    print()
    
    # Setup
    setup_directories()
    
    # Paths for temporary images
    temp_img1 = os.path.join(TEMP_DIR, "temp1.jpg")
    temp_img2 = os.path.join(TEMP_DIR, "temp2.jpg")
    
    # Capture initial reference frame
    print("Capturing initial reference frame...")
    if not capture_image(temp_img1):
        print("Failed to capture initial frame. Exiting.")
        return
    
    prev_frame = cv2.imread(temp_img1)
    if prev_frame is None:
        print("Failed to read initial frame. Exiting.")
        return
    
    print("✓ Ready! Monitoring for motion...\n")
    
    motion_count = 0
    check_count = 0
    
    try:
        while True:
            # Capture new frame
            if not capture_image(temp_img2):
                print("Failed to capture frame, skipping...")
                time.sleep(CHECK_INTERVAL)
                continue
            
            current_frame = cv2.imread(temp_img2)
            if current_frame is None:
                print("Failed to read frame, skipping...")
                time.sleep(CHECK_INTERVAL)
                continue
            
            # Detect motion
            motion_detected, area, diff_img = detect_motion(prev_frame, current_frame)
            
            check_count += 1
            
            if motion_detected:
                motion_count += 1
                save_motion_image(current_frame, area)
                
                # Optional: Save comparison images for debugging
                if SAVE_COMPARISON_IMAGES:
                    debug_path = os.path.join(TEMP_DIR, f"debug_{datetime.now().strftime('%H%M%S')}.jpg")
                    cv2.imwrite(debug_path, diff_img)
                
                # Cooldown period
                time.sleep(COOLDOWN_PERIOD)
            else:
                # Show progress indicator
                if check_count % 10 == 0:
                    print(f"Monitoring... (checks: {check_count}, motion events: {motion_count})")
            
            # Update reference frame
            prev_frame = current_frame.copy()
            
            # Wait before next check
            time.sleep(CHECK_INTERVAL)
    
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Motion detection stopped")
        print(f"Total checks: {check_count}")
        print(f"Motion events detected: {motion_count}")
        print("=" * 60)
    
    except Exception as e:
        print(f"\nError in main loop: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

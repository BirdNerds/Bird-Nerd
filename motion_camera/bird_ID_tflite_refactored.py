#!/usr/bin/env python3
"""
Bird Detection and Classification using TensorFlow Lite
Combines motion detection with on-device ML classification
Based on: https://mikesml.com/2021/05/16/image-recognition-on-the-edge-tflite-on-raspberry-pi/
"""

import warnings  # suppress noisy numpy warnings
warnings.filterwarnings("ignore", message="The value of the smallest subnormal for <class 'numpy.float64'> type is zero.", category=UserWarning, module="numpy.core.getlimits")
import os, time, subprocess, cv2, numpy as np  # imports
from datetime import datetime  # timestamping
from pathlib import Path

# configuration
MOTION_DIR = "/home/tocila/Documents/Motion_Camera/images"
CLASSIFIED_DIR = "/home/tocila/Documents/Motion_Camera/classified"
UNKNOWN_DIR = "/home/tocila/Documents/Motion_Camera/unknown"
TEMP_DIR = "/tmp/bird_feeder"
MODEL_PATH = "/home/tocila/Documents/Motion_Camera/models/bird_classifier.tflite"
LABELS_PATH = "/home/tocila/Documents/Motion_Camera/models/labels.txt"
CONFIDENCE_THRESHOLD = 0.6  # min confidence to mark classified
MOTION_THRESHOLD = 40  # frame diff threshold
MIN_AREA = 2000  # min contour area to consider motion
BLUR_SIZE = 21  # gaussian blur kernel
LARGE_MOVE_RATIO = 0.6  # skip classify if bbox covers this fraction of frame
BBOX_PADDING = 1.25  # bbox padding multiplier for crops
LAP_VAR_THRESHOLD = 12.0  # laplacian variance threshold for texture check
ASPECT_RATIO_RANGE = (0.08, 12.0)  # acceptable crop aspect ratios
IMAGE_WIDTH, IMAGE_HEIGHT = 1920, 1080  # capture size
CAPTURE_TIMEOUT = 2000  # camera timeout (ms)
CHECK_INTERVAL = 0.5  # main loop sleep
COOLDOWN_PERIOD = 2.0  # after detection sleep
IMAGE_PREFIX = "bird-"  # saved filename prefix
IGNORE_LABELS = {'background','person','cat','dog','car','truck'}  # ignore set

# try to import tflite runtime falling back to tensorflow.lite
try:
    import tflite_runtime.interpreter as tflite  # prefer lightweight runtime
except Exception:
    import tensorflow as tf
    tflite = tf.lite  # type: ignore

def ensure_dirs(): Path(d).mkdir(parents=True, exist_ok=True) or None  # helper for loop below
for d in (MOTION_DIR, CLASSIFIED_DIR, UNKNOWN_DIR, TEMP_DIR): ensure_dirs()

def now_ts(fmt="%Y%m%d_%H%M%S"): return datetime.now().strftime(fmt)  # timestamp helper

def load_labels(path):  # load label file into list
    try:
        return [line.strip() for line in open(path, 'r', encoding='utf-8').read().splitlines()]
    except Exception:
        return []

def load_model(model_path):
    try:
        interp = tflite.Interpreter(model_path=model_path)  # create interpreter
    except Exception:
        return None
    interp.allocate_tensors()
    return interp

def capture_image(path):  # capture using rpicam-still; return True if file exists
    try:
        subprocess.run(['rpicam-still','-o',path,'-t',str(CAPTURE_TIMEOUT),'--width',str(IMAGE_WIDTH),'--height',str(IMAGE_HEIGHT),'-n'],
                       capture_output=True, text=True, timeout=5)
        return os.path.exists(path)
    except Exception:
        return False

def detect_motion_bboxes(frame1, frame2):  # return total area and list of bboxes for motion
    g1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # grayscale
    g2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    g1 = cv2.GaussianBlur(g1, (BLUR_SIZE,BLUR_SIZE), 0)  # blur to reduce noise
    g2 = cv2.GaussianBlur(g2, (BLUR_SIZE,BLUR_SIZE), 0)
    delta = cv2.absdiff(g1, g2)  # absolute diff
    th = cv2.threshold(delta, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    th = cv2.dilate(th, None, iterations=2)
    cnts, _ = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = 0
    bboxes = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a > MIN_AREA:
            total_area += a
            bboxes.append(cv2.boundingRect(c))
    return total_area, bboxes

def crop_pad(frame, bbox):  # pad bbox and clip to frame bounds
    x,y,w,h = bbox
    cx,cy = x + w//2, y + h//2
    nw,nh = int(w*BBOX_PADDING), int(h*BBOX_PADDING)
    x1, y1 = max(0, cx - nw//2), max(0, cy - nh//2)
    x2, y2 = min(frame.shape[1], cx + nw//2), min(frame.shape[0], cy + nh//2)
    return frame[y1:y2, x1:x2]

def robust_softmax(logits):  # stable softmax with nan/inf handling
    l = np.array(logits, dtype=np.float64)
    l = np.nan_to_num(l, nan=-1e9, posinf=1e9, neginf=-1e9)
    l = np.clip(l, -100.0, 100.0)
    s = l - np.max(l)  # shift
    e = np.exp(s)
    den = np.sum(e)
    if den == 0 or not np.isfinite(den):
        return np.zeros_like(e)
    return e/den

def preprocess_for_model(img, input_shape, dtype):  # resize, convert, normalize
    h,w = input_shape[1], input_shape[2]
    r = cv2.resize(img, (w,h))
    r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
    if dtype == np.float32: r = r.astype(np.float32) / 255.0
    return np.expand_dims(r, axis=0)

def classify_crop(interp, input_details, output_details, labels, crop):  # run tflite classify
    data = preprocess_for_model(crop, input_details[0]['shape'], input_details[0]['dtype'])
    interp.set_tensor(input_details[0]['index'], data)
    interp.invoke()
    out = interp.get_tensor(output_details[0]['index'])[0]
    probs = robust_softmax(out)
    idx = int(np.argmax(probs))
    conf = float(probs[idx]) if probs.size else 0.0
    label = labels[idx] if idx < len(labels) else f"class_{idx}"
    top3_idx = np.argsort(probs)[-3:][::-1] if probs.size else []
    top3 = [(labels[i] if i < len(labels) else f"class_{i}", float(probs[i])) for i in top3_idx]
    return label, conf, top3

def save_with_metadata(frame, label, conf, top3, dest_dir):  # save image + metadata
    ts = now_ts()
    fname = f"{IMAGE_PREFIX}{label}_{ts}_{conf:.2f}.jpg"
    path = os.path.join(dest_dir, fname)
    cv2.imwrite(path, frame)
    meta = path.replace('.jpg','.txt')
    with open(meta, 'w', encoding='utf-8') as f:
        f.write(f"Top prediction: {label} ({conf:.2%})\n")
        f.write("Top 3:\n")
        for i,(lbl,c) in enumerate(top3,1): f.write(f"{i}. {lbl}: {c:.2%}\n")
    return path

def should_classify_crop(crop):  # quick checks: aspect ratio and laplacian variance
    if crop.size == 0: return False, "empty"
    h,w = crop.shape[:2]
    ar = (w/float(h)) if h>0 else 0.0
    if not (ASPECT_RATIO_RANGE[0] <= ar <= ASPECT_RATIO_RANGE[1]): return False, f"aspect:{ar:.2f}"
    lap = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    if lap < LAP_VAR_THRESHOLD: return False, f"lap_var:{lap:.2f}"
    return True, f"lap_var:{lap:.2f}"

def main():  # main loop
    labels = load_labels(LABELS_PATH)  # load labels once
    interp = load_model(MODEL_PATH)  # load model or None
    if interp:
        input_details = interp.get_input_details()
        output_details = interp.get_output_details()
        print(f"Model loaded: {MODEL_PATH} shape={input_details[0]['shape']}")
    else:
        print("Model not loaded; running motion-only mode")
    print(f"Pre-classify thresholds: lap_var >= {LAP_VAR_THRESHOLD}, aspect_ratio in {ASPECT_RATIO_RANGE}")
    temp1 = os.path.join(TEMP_DIR, "temp1.jpg"); temp2 = os.path.join(TEMP_DIR, "temp2.jpg")
    if not capture_image(temp1): print("Failed to capture initial frame"); return
    prev = cv2.imread(temp1)
    if prev is None: print("Failed to read initial frame"); return
    print("Ready! Monitoring for birds...")
    checks = detections = 0
    try:
        while True:
            if not capture_image(temp2): time.sleep(CHECK_INTERVAL); continue
            cur = cv2.imread(temp2); 
            if cur is None: time.sleep(CHECK_INTERVAL); continue
            checks += 1
            area, bboxes = detect_motion_bboxes(prev, cur)
            if bboxes:
                tstr = datetime.now().strftime("%H:%M:%S")
                print(f"{tstr} Motion detected area={area:.0f}")
                # choose largest bbox and possibly skip classification if huge
                bx = max(bboxes, key=lambda b: b[2]*b[3])
                bw,bh = bx[2], bx[3]
                if (bw*bh) / float(IMAGE_WIDTH*IMAGE_HEIGHT) >= LARGE_MOVE_RATIO:
                    print(f"{tstr} Skipping classification (large region)") 
                    ts = now_ts(); fname = f"{IMAGE_PREFIX}{ts}.jpg"; cv2.imwrite(os.path.join(MOTION_DIR,fname), cur); detections += 1
                else:
                    crop = crop_pad(cur, bx)
                    ok, reason = should_classify_crop(crop)
                    if not ok:
                        print(f"{tstr} Skipping classification ({reason})"); ts = now_ts(); cv2.imwrite(os.path.join(MOTION_DIR,f"{IMAGE_PREFIX}{ts}.jpg"), cur); detections += 1
                    elif interp:
                        label, conf, top3 = classify_crop(interp, input_details, output_details, labels, crop)
                        dest = CLASSIFIED_DIR if conf >= CONFIDENCE_THRESHOLD and label.lower() not in IGNORE_LABELS else UNKNOWN_DIR
                        save_with_metadata(cur, label, conf, top3, dest)
                        detections += 1
                    else:
                        ts = now_ts(); cv2.imwrite(os.path.join(MOTION_DIR,f"{IMAGE_PREFIX}{ts}.jpg"), cur); detections += 1
                time.sleep(COOLDOWN_PERIOD)
            else:
                if checks % 10 == 0: print(f"Monitoring... (checks: {checks}, detections: {detections})")
            prev = cur.copy(); time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        print("Detection stopped") 
        print(f"Total checks: {checks} Detections: {detections}")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__": main()

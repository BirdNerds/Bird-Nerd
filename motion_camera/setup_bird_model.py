#!/usr/bin/env python3
"""
Download Google AIY Birds model using Kaggle Hub
"""

import os
import sys
from pathlib import Path

print("=" * 60)
print("Google AIY Birds Model - Kaggle Download")
print("=" * 60)

# Check if kagglehub is installed
try:
    import kagglehub
except ImportError:
    print("\n❌ kagglehub not installed")
    print("\nInstalling kagglehub...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
    import kagglehub
    print("✓ kagglehub installed")

# Configuration
MODEL_DIR = os.path.expanduser("~/Documents/models")
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

print(f"\nModel directory: {MODEL_DIR}")
print(f"Source: Kaggle - Google AIY Vision Classifier Birds V1")

# Download model
print(f"\n1. Downloading model from Kaggle...")
print(f"   This may take a few minutes...")

try:
    # Download the model
    path = kagglehub.model_download("google/aiy/tensorFlow1/vision-classifier-birds-v1")
    
    print(f"   ✓ Model downloaded to: {path}")
    
    # List files in the downloaded directory
    print(f"\n   Files in model directory:")
    import glob
    all_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            filepath = os.path.join(root, file)
            all_files.append(filepath)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"   - {os.path.relpath(filepath, path)} ({size_mb:.1f} MB)")
    
    # Find TFLite model
    tflite_files = [f for f in all_files if f.endswith('.tflite')]
    
    if tflite_files:
        source_model = tflite_files[0]
        target_model = os.path.join(MODEL_DIR, "bird_classifier.tflite")
        
        # Copy to our models directory
        import shutil
        shutil.copy2(source_model, target_model)
        print(f"\n   ✓ Model copied to: {target_model}")
        
        MODEL_PATH = target_model
    else:
        print(f"\n   ⚠️  No .tflite file found. Using downloaded path directly.")
        MODEL_PATH = path
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    print("\n   Make sure you have Kaggle credentials set up:")
    print("   1. Create account at kaggle.com")
    print("   2. Go to Account -> API -> Create New Token")
    print("   3. Save kaggle.json to ~/.kaggle/kaggle.json")
    print("   4. Run: chmod 600 ~/.kaggle/kaggle.json")
    exit(1)

# Download labels
print(f"\n2. Downloading labels...")
LABELS_URL = "https://raw.githubusercontent.com/google-coral/test_data/master/inat_bird_labels.txt"
LABELS_PATH = os.path.join(MODEL_DIR, "labels.txt")

try:
    import urllib.request
    urllib.request.urlretrieve(LABELS_URL, LABELS_PATH)
    
    with open(LABELS_PATH, 'r') as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"   ✓ Downloaded: {LABELS_PATH}")
    print(f"   ✓ Bird species: {len(labels)}")
    
    print(f"\n   Sample species:")
    for label in labels[:5]:
        print(f"   - {label}")
    if len(labels) > 5:
        print(f"   ... and {len(labels) - 5} more")
        
except Exception as e:
    print(f"   ⚠️  Could not download labels: {e}")
    print(f"   You may need to create this manually")

# Test the model
print(f"\n3. Testing model...")
try:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite
    
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"   ✓ Model loaded successfully!")
    print(f"   Input shape: {input_details[0]['shape']}")
    print(f"   Input type: {input_details[0]['dtype']}")
    print(f"   Output shape: {output_details[0]['shape']}")
    
except ImportError:
    print(f"   ⚠️  TensorFlow Lite not installed")
    print(f"   Install with: pip install tflite-runtime")
except Exception as e:
    print(f"   ⚠️  Error testing model: {e}")

# Final instructions
print("\n" + "=" * 60)
print("✓ Setup Complete!")
print("=" * 60)
print(f"Model: {MODEL_PATH}")
print(f"Labels: {LABELS_PATH}")
print(f"\nNext steps:")
print("1. Update bird_detection_tflite.py paths if needed")
print("2. Run: python3 bird_detection_tflite.py")
print("3. Point camera at birds!")
print("=" * 60)

#!/usr/bin/env python3
"""
Convert TensorFlow SavedModel to TFLite format
"""

import os
import tensorflow as tf
import numpy as np
from pathlib import Path

print("=" * 60)
print("Converting SavedModel to TFLite")
print("=" * 60)

# Paths
SAVED_MODEL_PATH = "/home/tocila/.cache/kagglehub/models/google/aiy/tensorFlow1/vision-classifier-birds-v1/1"
OUTPUT_DIR = os.path.expanduser("~/Documents/models")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "bird_classifier.tflite")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print(f"\nSource: {SAVED_MODEL_PATH}")
print(f"Output: {OUTPUT_PATH}")

print("\n1. Loading SavedModel...")
try:
    # Load the SavedModel
    model = tf.saved_model.load(SAVED_MODEL_PATH)
    print("   ✓ Model loaded")
    
    # Get the concrete function
    concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    print(f"   ✓ Found signature")
    
    # Print input/output info
    print(f"\n   Input info:")
    for key, value in concrete_func.structured_input_signature[1].items():
        print(f"   - {key}: {value}")
    
    print(f"\n   Output info:")
    for key, value in concrete_func.structured_outputs.items():
        print(f"   - {key}: {value}")
    
except Exception as e:
    print(f"   ❌ Error loading model: {e}")
    print("\n   The model format may not be compatible.")
    print("   Let's try a simpler approach - downloading a working TFLite model...")
    exit(1)

print("\n2. Converting to TFLite...")
try:
    # Create converter
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_PATH)
    
    # Set optimization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert
    print("   Converting (this may take 2-3 minutes)...")
    tflite_model = converter.convert()
    
    # Save
    with open(OUTPUT_PATH, 'wb') as f:
        f.write(tflite_model)
    
    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"   ✓ Converted successfully!")
    print(f"   ✓ Saved to: {OUTPUT_PATH}")
    print(f"   ✓ Size: {size_mb:.1f} MB")
    
except Exception as e:
    print(f"   ❌ Conversion failed: {e}")
    print("\n   The TensorFlow 1.x model may not convert easily.")
    print("   Recommendation: Use a pre-made TFLite model instead")
    exit(1)

print("\n3. Testing converted model...")
try:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite
    
    interpreter = tflite.Interpreter(model_path=OUTPUT_PATH)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"   ✓ Model works!")
    print(f"   Input shape: {input_details[0]['shape']}")
    print(f"   Output shape: {output_details[0]['shape']}")
    
except Exception as e:
    print(f"   ⚠️  Test failed: {e}")

print("\n" + "=" * 60)
print("✓ Conversion Complete!")
print("=" * 60)
print(f"TFLite model: {OUTPUT_PATH}")
print("\nNext: Run python3 bird_detection_tflite.py")
print("=" * 60)

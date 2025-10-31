# Bird-Nerd
CS 396/398: Senior Project - An IoT Bird Feeder with AI-Powered Species Identification

An automated bird identification system that uses motion detection and on-device machine learning to recognize visiting birds at your feeder. Built to run efficiently on Raspberry Pi 4 with minimal resources.

## Project Description

Bird Nerd combines computer vision with IoT monitoring to automatically identify bird species visiting a backyard feeder. The system uses motion detection to trigger image capture, then classifies birds using Google's AIY Birds V1 TensorFlow Lite model (964 species). Designed for deployment on resource-constrained edge devices, our system runs on a Raspberry Pi 4 with only 1GB RAM.

Future features include automatic social media posting, database storage of sightings, and notifications for low seed levels or unwanted visitors (squirrels).

## Repository Structure

### `motion_camera/`
Production implementation using TensorFlow Lite for efficient on-device inference. Combines motion detection with real-time classification, optimized to run on Raspberry Pi 4.

**Key files:**
- `bird_ID_tflite.py` - Main detection and classification system
- `bird_ID_tflite_refactored.py` - Improved version (in progress)
- `setup_bird_model.py` - Downloads and configures the Google AIY model
- `convert_to_tflite.py` - Model conversion utilities
- `models/` - Stores TFLite model and label files
- `classified/` - High-confidence bird identifications
- `unknown/` - Low-confidence or filtered detections
- `images/` - General motion-detected images

### `toy_model/`
Initial prototype using Ollama's llava:7b vision model for bird identification. This was a proof-of-concept to test accuracy and validate the project approach. Too slow and resource-intensive for deployment on Raspberry Pi (30+ seconds per image), but useful for initial testing on more powerful hardware.

**Key files:**
- `bird_nerd.py` - CLI tool for identifying birds from image files
- `test_pictures/` - Sample images for testing

## Quick Start

*Documentation in progress - setup instructions coming soon*

## Hardware Requirements
- Raspberry Pi 4 (1GB+ RAM)
- Raspberry Pi Camera Module (we're using X000VGJ8BL)
- MicroSD card (16GB+ recommended)
- Bird feeder (installation planned)

## Software Requirements
- Python 3.13
- TensorFlow Lite Runtime
- OpenCV
- NumPy

## Current Status
✅ Motion detection working  
✅ TFLite classification working  
✅ Efficient operation on 1GB RAM (~280MB usage)  
🚧 Social media integration (in progress)  
🚧 Database storage (in progress)  
🚧 Outdoor deployment (planned)

## Authors
Jacob Tocila and Sam Lamsma  
Calvin University, Fall 2025 - Spring 2026

## Acknowledgments
Inspired by Mike Schultz's work on [TensorFlow Lite edge deployment](https://mikesml.com/2021/05/16/image-recognition-on-the-edge-tflite-on-raspberry-pi/)

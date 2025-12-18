# Bird-Nerd
CS 396/398: Senior Project - An IoT Bird Feeder with AI-Powered Species Identification 

## Vision
Bird Nerd provides backyard bird watchers with an intelligent, automated tool that brings joy and discovery to nature enthusiasts. By utilizing AI-powered species identification with IoT monitoring, the system automatically recognizes visiting birds at the bird feeder, takes a snapshot, and shares images of sightings on social media. 
This can help keep the birding community engaged with real-time wildlife updates. 

The platform also serves as a practical companion for bird feeding, notifying the owner when uninvited guests (namely squirrels) appear, or when seed levels run low. Ultimately, Bird Nerd makes it effortless to connect with nature, learn about local bird species, and share the wonder of backyard birding with others.

## Authors
Jacob Tocila: 25, A senior at Calvin University majoring in computer science with minors in data science and Dutch studies. He is interested in information security, baking, and puzzles.

Sam Lamsma: 21, A senior at Calvin University, majoring in computer science. He is interested in Cybersecurity and also throws for the Calvin University Track and Field Team.

Professor Derek C. Schuurman, current chair of the Calvin computer science department and the advisor for the project.  

Calvin University, Fall 2025 - Spring 2026

## Code
The source code for our project can be found [here](https://github.com/BirdNerds/Bird-Nerd).

## Deliverables

### Report
The report for our project can be found [here](https://docs.google.com/document/d/1ejYF54ZocJHBa88cmade4xO01HcAqiNkbfw3FI-FXoc/edit?tab=t.aa46uj7emtqj).

### Department of Computer Science website
To learn more about Calvin University’s CS department, visit [this link](https://calvin.edu/academics/school-stem/computer-science).

### Presentation
To visit the presentation slides we gave for this assignment, visit [this link](https://docs.google.com/presentation/d/1DoOasrh5okXjUorbduDV5vdW57nHWcixiS5i9uhj74Q/edit?usp=sharing). 

### Project Description

Bird Nerd combines computer vision with IoT monitoring to automatically identify bird species visiting a backyard feeder. The system uses motion detection to trigger image capture, then classifies birds using Google's AIY Birds V1 TensorFlow Lite model (964 species). Designed for deployment on resource-constrained edge devices, our system runs on a Raspberry Pi 4 with only 1GB RAM.

Future features include automatic social media posting, database storage of sightings, and notifications for low seed levels or unwanted visitors (squirrels).

## Repository Structure

### `motion_camera/`
Production implementation using TensorFlow Lite for efficient on-device inference. Combines motion detection with real-time classification, optimized to run on Raspberry Pi 4.

**Key files:**
- `bird_ID_tflite.py` - Main detection and classification system
- `setup_bird_model.py` - Downloads and configures the Google AIY model
- `convert_to_tflite.py` - Model conversion utilities
- `models/` - Stores TFLite model and label files
- `images/` - General motion-detected images
- `unidentified/` - Images not sent to the GitHub/website automatically. Needs manual review

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

## Acknowledgments
Inspired by Mike Schultz's work on [TensorFlow Lite edge deployment](https://mikesml.com/2021/05/16/image-recognition-on-the-edge-tflite-on-raspberry-pi/)

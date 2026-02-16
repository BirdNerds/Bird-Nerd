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
The source code for our project can be found [on our GitHub](https://github.com/BirdNerds/Bird-Nerd).

## Deliverables

### Report
The report for our project can be found [in our Google Docs](https://docs.google.com/document/d/1ejYF54ZocJHBa88cmade4xO01HcAqiNkbfw3FI-FXoc/edit?tab=t.aa46uj7emtqj) documentation.

### Department of Computer Science website
To learn more about our school's program, visit [Calvin University's CS department home page](https://calvin.edu/academics/school-stem/computer-science).

### Presentation
To visit the presentation slides we gave for this project, visit [our Google Slides](https://docs.google.com/presentation/d/1DoOasrh5okXjUorbduDV5vdW57nHWcixiS5i9uhj74Q/edit?usp=sharing). 

### Project Description

Bird Nerd combines computer vision with IoT monitoring to automatically identify bird species visiting a backyard feeder. The system uses motion detection to trigger image capture, then classifies birds using Google's AIY Birds V1 TensorFlow Lite model (964 species). Designed for deployment on resource-constrained edge devices, our system runs on a Raspberry Pi 4 with only 1GB RAM.

## Repository Structure

### `motion_camera/`
Production implementation using TensorFlow Lite for efficient on-device inference. Combines motion detection with real-time classification, optimized to run on Raspberry Pi 4.

**Key files:**
- `bird_ID_tflite.py` - Main detection and classification system
- `setup_bird_model.py` - Downloads and configures the Google AIY model
- `convert_to_tflite.py` - Model conversion utilities
- `firebase_helper.py` - Manages Firestore client for live database reads/writes
- `models/` - Stores TFLite model and label files
- `images/` - Local copies of images taken by `bird_ID_tflite.py`
- `venv/` - Virtual environment used for Python libraries and such

### `website/`
All files for website hosting

**Key files:**
- `index.html` - HTML for the website, hosted on [https://students.cs.calvin.edu/~jt42/](https://students.cs.calvin.edu/~jt42/)
- `styles.css` - A .css file for the main page
- `Robin_PCB.png` - Bird_Nerd logo image
- `Robin_PCB_Favicon.png` - Favicon icon for website

### `toy_model/`
Initial prototype using Ollama's llava:7b vision model for bird identification. This was a proof-of-concept to test accuracy and validate the project approach. Too slow and resource-intensive for deployment on Raspberry Pi (30+ seconds per image), but useful for initial testing on more powerful hardware. This is still in the Repo for fun.

**Key files:**
- `bird_nerd.py` - CLI tool for identifying birds from image files
- `test_pictures/` - Sample images for testing

## Quick Start

*Documentation in progress - setup instructions coming soon. There are rough steps to setting a virtual environment to run bird_ID_tflite.py on a Raspberry Pi in the Google Docs, under "To create virtual environment" section. This will be refined further in the future.*

## Hardware Requirements
- Raspberry Pi 4 (1GB+ RAM)
- Raspberry Pi Camera Module (we're using X000VGJ8BL)
- MicroSD card (16GB+ recommended)
- A bird feeder. We bought [this bird feeder](https://www.tractorsupply.com/tsc/product/royal-wing-suet-combo-wooden-bird-feeder-with-galvanized-roof-2469759) from Tractor Supply Co. since it's relatively inexpensive 

## Software Requirements
- Python 3.13
- TensorFlow Lite Runtime
- Pip
- OpenCV
- NumPy
- Firebase-admin
- dotenv
- tzlocal

## Current Status
✅ Motion detection working  
✅ TFLite classification working  
✅ Efficient operation on 1GB RAM Pi (~280MB usage)  
✅ Database storage  
🚧 Website integration (in progress)  
🚧 Outdoor deployment (coming soon)

## Acknowledgments
Inspired by Mike Schultz's work on [TensorFlow Lite edge deployment](https://mikesml.com/2021/05/16/image-recognition-on-the-edge-tflite-on-raspberry-pi/)

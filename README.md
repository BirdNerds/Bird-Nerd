# Bird-Nerd
CS 396/398: Senior Project - An IoT Bird Feeder with AI-Powered Species Identification

> **The live project website is at [https://students.cs.calvin.edu/~jt42/](https://students.cs.calvin.edu/~jt42/)** - Check out real-time sightings from the feeder here.

---

## Vision

Bird Nerd is a real-time backyard bird identification system built with a Raspberry Pi 4, a window-mounted camera, and a TensorFlow Lite classifier running entirely on-device. When motion is detected at the feeder, a short burst of frames is captured, classified by species, and uploaded to a live website automatically. No cloud GPU or manual review.

The classifier uses a multi-frame voting system. Several frames from each visit are independently classified using Google's AIY Birds V1 model (964 species), and the winning species is determined by plurality, with confidence averaged across agreeing frames. This makes the system substantially more reliable than single-frame inference, especially for fast-moving or partially obscured birds.

Jacob is currently using this [Going Green window feeder](https://happybirdwatcher.com/products/going-green-window-feeder). Sightings are logged to Firebase and displayed on the website in real time. Note: bird species identification is performed by a machine learning model and may be incorrect. Confidence scores reflect model certainty, not biological accuracy.

---

## Authors

**Jacob Tocila** - A senior at Calvin University majoring in Computer Science with minors in Data Science and Dutch Studies. He is also working on a CompTIA Security+ certification at Calvin. He is interested in information security, birdwatching, IoT, and puzzles.

**Professor Derek C. Schuurman** - Current chair of the Calvin University Computer Science department and project advisor.

Calvin University, Fall 2025 – Spring 2026

---

## Links

- **Live website**: [https://students.cs.calvin.edu/~jt42/](https://students.cs.calvin.edu/~jt42/)
- **Source code**: [github.com/BirdNerds/Bird-Nerd](https://github.com/BirdNerds/Bird-Nerd)
- **Project report**: [Google Docs](https://docs.google.com/document/d/1ejYF54ZocJHBa88cmade4xO01HcAqiNkbfw3FI-FXoc/edit?tab=t.aa46uj7emtqj)
- **Presentation slides**: [Google Slides](https://docs.google.com/presentation/d/1DoOasrh5okXjUorbduDV5vdW57nHWcixiS5i9uhj74Q/edit?usp=sharing)
- **Calvin CS Department**: [calvin.edu/academics/school-stem/computer-science](https://calvin.edu/academics/school-stem/computer-science)

---

## Hardware

- Raspberry Pi 4 (1 GB RAM)
- Raspberry Pi Camera Module (X000VGJ8BL)
- [Going Green Window Bird Feeder](https://happybirdwatcher.com/products/going-green-window-feeder) by Woodlink
- MicroSD card (16 GB+)

---

## Repository Structure

```
Bird-Nerd/
├── motion_camera/          # Production system - runs on the Pi
│   ├── main.py
│   ├── config.py
│   ├── bird_classify.py
│   ├── frame_capture.py
│   ├── motion_detect.py
│   ├── gif_builder.py
│   ├── firebase_upload.py
│   ├── firebase_helper.py
│   ├── sighting_log.py
│   ├── setup_bird_model.py
│   ├── convert_to_tflite.py
│   ├── models/             # TFLite model + labels.txt (not committed)
│   ├── images/             # High-confidence sighting stills (not committed)
│   ├── unclear_images/     # Low-confidence sighting stills (not committed)
│   └── sightings.log       # Local text log (not committed)
└── website/                # Static frontend
    ├── index.html
    ├── styles.css
    ├── firebase_functions.js
    └── firebase_config.example.js
```

### `motion_camera/`

The production system that runs on the Raspberry Pi.

| File | Description |
|---|---|
| `main.py` | The main program - watches for motion, triggers photo capture, identifies the bird, and sends the result to Firebase. |
| `config.py` | All settings in one place: camera resolution, detection sensitivity, confidence thresholds, file paths, and timing. |
| `bird_classify.py` | Identifies the bird species by running the AI model across several frames from each visit and picking the most agreed-upon answer. |
| `frame_capture.py` | Controls the camera - takes photos for motion checking and records short bursts of frames when a bird arrives. |
| `motion_detect.py` | Watches a cropped section of the frame and flags when something moves that's the right size to be a bird. |
| `gif_builder.py` | Turns the burst of captured frames into an animated GIF and picks the clearest single frame as the thumbnail photo. |
| `firebase_upload.py` | Sends the sighting details, photo, and GIF to the cloud database so they appear on the website. |
| `firebase_helper.py` | Used to send dummy fake birds to the Firebase database for testing purposes. |
| `sighting_log.py` | Saves a plain-text record of each sighting to a local log file on the Pi. |
| `setup_bird_model.py` | Downloads the bird identification AI model and species label list from Kaggle. |
| `convert_to_tflite.py` | One-time utility for converting the original model file into the lightweight format the Pi uses. |
| `models/` | Directory for the TFLite model file and its labels (not committed to Git). |
| `.env` | Environment variables for Firebase credentials and other secrets (not committed to Git). |

### `website/`

A static site that reads sightings from Firebase in real time and displays them publicly.

| File | Description |
|---|---|
| `index.html` | Main page - displays live sightings with species name, confidence, and photo. |
| `styles.css` | Stylesheet for the main page. |
| `firebase_functions.js` | Queries Firestore for new sightings and updates the page in real time. |
| `firebase_config.example.js` | Template for the Firebase web app configuration (real config is gitignored). |
| `bird_chirp.wav` | Sound effect played when a new sighting is added to the page. |
| `Robin_PCB.png` | Main logo image, generated with ChatGPT. |
| `Robin_PCB_Favicon.png` | Custom favicon for the website. |

---

## Quick Start

See [`QUICKSTART.md`](QUICKSTART.md) for full setup instructions: includes camera setup, model download, and Firebase rules configuration.

---

## Acknowledgments

Inspired by Mike Schultz's work on [TensorFlow Lite edge deployment](https://mikesml.com/2021/05/16/image-recognition-on-the-edge-tflite-on-raspberry-pi/).
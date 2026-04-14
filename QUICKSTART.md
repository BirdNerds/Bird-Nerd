# Bird Nerd - Quick Start Guide

> **Bird Nerd** is an IoT bird feeder system that uses motion detection and on-device TensorFlow Lite inference to automatically identify bird species, log sightings to Firebase, and display them on a live website.

This Quick Start Guide is a tutorial to help you set up your own Bird Nerd system.

---

## Prerequisites

- Raspberry Pi 4 (1 GB+ RAM)
- Raspberry Pi Camera Module (tested with `X000VGJ8BL`)
- MicroSD card (16 GB+ recommended) with Raspberry Pi OS (64-bit)
- Python 3.13
- A [Firebase](https://firebase.google.com/) account (free Spark tier is sufficient)

---

## 1. Hardware Setup

Attach the camera module to the Raspberry Pi's CSI camera port and mount the Pi near your bird feeder with a clear line of sight. I have mine sitting on a windowsill. To confirm the camera is working before continuing, open a Python shell and run:

```python
from picamera2 import Picamera2
cam = Picamera2()
cam.start()
cam.capture_file("test.jpg")
cam.stop()
```

If `test.jpg` is saved without errors, you're good to go.

---

## 2. Clone the Repository

```bash
git clone https://github.com/BirdNerds/Bird-Nerd.git
cd Bird-Nerd
```

---

## 3. Set Up the Python Environment

```bash
cd motion_camera
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

The key dependencies are:

| Package | Version | Purpose |
|---|---|---|
| `tflite-runtime` | 2.14.0 | Lightweight TensorFlow Lite runtime for running the bird classifier |
| `tensorflow` | 2.20.0 | Full TensorFlow fallback if tflite-runtime is unavailable |
| `opencv-python-headless` | 4.13.0.92 | Frame capture, motion detection, and image processing |
| `numpy` | 1.26.4 | Numerical array operations used throughout |
| `pillow` | 12.1.1 | GIF assembly |
| `firebase-admin` | 7.1.0 | Writing sightings to Firestore and uploading photos to Storage |
| `python-dotenv` | 1.2.1 | Loading credentials from the `.env` file |
| `pytz` | 2025.2 | Timezone handling for sighting timestamps |
| `tzlocal` | 5.3.1 | Detects the system's local timezone |
| `picamera2` | 0.3.34 | Camera interface for the Raspberry Pi camera module |

> **Note:** I highly recommend using `tflite-runtime` on the Pi for its smaller footprint. If it is unavailable for your platform, you can use `tensorflow` instead. The code should switch to it automatically.

---

## 4. Download the Bird Classification Model

Bird Nerd uses [Google's AIY Vision Classifier Birds V1](https://www.kaggle.com/models/google/aiy/tensorFlow1/vision-classifier-birds-v1), which can identify 964 species. A helper script handles the download.

You will need a Kaggle account and API token:

1. Create a free account at [kaggle.com](https://www.kaggle.com)
2. Go to **Account → Settings → API → Create New Token** - this downloads `kaggle.json`
3. Place it on the Pi:

```bash
mkdir -p ~/.kaggle
mv /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Then run the setup script:

```bash
python3 setup_bird_model.py
```

This downloads the model to `motion_camera/models/bird_classifier.tflite` and the species labels to `motion_camera/models/labels.txt`.

> **Note:** If you already have a different `.tflite` model file you'd like to use instead, drop it into `motion_camera/models/` and name it `bird_classifier.tflite`. Also place the labels file alongside it as `labels.txt`.

---

## 5. Firebase Setup

Bird Nerd uses Firebase for two things: **Firestore** (the sightings database) and **Firebase Storage** (storing bird photos and GIFs).

### 5a. Create a Firebase Project

1. Go to the [Firebase Console](https://console.firebase.google.com/) and click **Add project**
2. Give it a name (e.g., `bird-nerd`) and follow the prompts
3. Once created, go to **Build → Firestore Database → Create database**
   - Choose **Start in production mode**
   - Pick the region closest to you
4. Go to **Build → Storage → Get started** and follow the same steps

### 5b. Security Rules

In the Firebase Console under **Firestore → Rules**, paste the following and click **Publish**:

```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /heartbeat/{doc} {
      allow read: if true;
      allow write: if false;  // only Python backend writes
    }
    match /sightings/{sightingId} {
      allow read: if true;
      allow delete: if request.auth != null && request.auth.token.email.matches('.*@birdnerd\\.local');
      allow create: if false;
    }
  }
}
```

This allows anyone to read sightings publicly and blocks all writes from the browser. Only your Raspberry Pi's service account can add new sightings.

Do the same under **Storage → Rules**:

```
rules_version = '2';
service firebase.storage {
  match /b/{bucket}/o {
    match /sightings/{imageId} {
      allow read: if true;          // anyone can view bird photos
      allow write: if false;        // only the Pi's service account writes
    }
  }
}
```

### 5c. Generate a Service Account Key

The Pi needs admin credentials to write to Firebase.

1. In the Firebase Console, go to **Project Settings → Service accounts**
2. Click **Generate new private key** and download the JSON file
3. On the Pi, create the credentials directory and move the key there:

```bash
mkdir -p motion_camera/.credentials
mv ~/Downloads/your-key.json motion_camera/.credentials/bird-nerd-firebase-adminsdk.json
chmod 600 motion_camera/.credentials/bird-nerd-firebase-adminsdk.json
```

The downloaded JSON will look something like this:

```json
{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "abc123...",
  "private_key": "-----BEGIN RSA PRIVATE KEY-----\n...\n-----END RSA PRIVATE KEY-----\n",
  "client_email": "firebase-adminsdk-xxxxx@your-project-id.iam.gserviceaccount.com",
  "client_id": "123456789",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/..."
}
```

> **Note:** ⚠️ **Never commit this file.** It is gitignored via `.credentials/` and `*-adminsdk*.json`. Treat it like a password.

### 5d. Create the `.env` File

```bash
cd motion_camera
cp .env.example .env
```

Edit `motion_camera/.env` and fill in your values:

```
FIREBASE_CREDENTIALS_PATH=./.credentials/bird-nerd-firebase-adminsdk.json
FIREBASE_STORAGE_BUCKET=your-project-id.firebasestorage.app
```

Replace `your-project-id` with your actual Firebase project ID, found in **Firebase Console → Project Settings**.

### 5e. Test the Firebase Connection

```bash
python3 firebase_helper.py
```

A successful run adds a test sighting to Firestore and prints the document ID. You can verify it appeared in the Firebase Console under **Firestore → sightings**. You will also see a lovely "dummy fake bird" image appear on the website if you have it running locally (see next section).

---

## 6. Run the Detector

```bash
cd motion_camera
source venv/bin/activate
python3 main.py
```

The program will:
- Check for motion every 0.4 seconds
- When motion is detected, record a burst of frames and classify the bird species
- Save a photo to `images/` (high confidence) or `unclear_images/` (low confidence)
- Log the sighting to `sightings.log` and upload it to Firebase

Stop it with `Ctrl+C`.

---

## 7. Website

The `website/` directory contains a static site that reads sightings from Firebase in real time.

### 7a. Firebase Config File

Copy the example config and fill in your values:

```bash
cd website
cp firebase_config.example.js firebase_config.js
```

Edit `website/firebase_config.js`:

```javascript
export const firebaseConfig = {
    apiKey: "YOUR_API_KEY",
    authDomain: "your-project-id.firebaseapp.com",
    projectId: "your-project-id",
    storageBucket: "your-project-id.firebasestorage.app",
    messagingSenderId: "YOUR_MESSAGING_SENDER_ID",
    appId: "YOUR_APP_ID"
};

// Displayed as "Live Bird Sightings from {ownerName}'s Backyard"
export const siteConfig = {
    ownerName: "your-name-here",
};
```

Find the Firebase values in the Firebase Console under **Project Settings → Your apps → Web app → SDK setup and configuration**. If you haven't registered a web app yet, click **Add app → Web** and follow the prompts. Replace `your-name-here` with your name or whatever you'd like the site to say.

> **Note:** ⚠️ `firebase_config.js` is gitignored - never commit it. Only `firebase_config.example.js` belongs in the repo.

### 7b. Running Locally

```bash
cd website
python3 -m http.server 8080
# Open http://localhost:8080 in your browser
```

The site reads directly from Firebase, so all sightings appear even when the Pi is offline.

---

## Key Configuration Options

All tunable parameters live in `motion_camera/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.60` | Below this, a sighting is saved locally but not uploaded to Firebase. |
| `NO_MOTION_TIMEOUT` | `4.0s` | How long to wait with no new frames before deciding the bird has left. |
| `MAX_VISIT_DURATION` | `10.0s` | Hard cap on how long a single visit's burst capture can run. |
| `IDLE_CHECK_INTERVAL` | `0.4s` | How often to check for motion when no bird is present. |
| `MOTION_THRESHOLD` | `75` | How different two frames need to be (per pixel) to count as motion. |
| `MIN_CONTOUR_AREA` | `12000` | Minimum size of a moving region (in pixels²) to count as a bird. |
| `LARGE_MOTION_RATIO` | `0.35` | If motion covers more than this fraction of the frame, it's ignored as a lighting change. |
| `LAP_VAR_THRESHOLD` | `18.0` | Skips classification if the image is too smooth to plausibly contain a bird. |
| `BURST_FPS` | `6` | Frames per second during a bird visit recording. |
| `CLASSIFY_EVERY_N_FRAMES` | `4` | Run the classifier on every Nth burst frame to keep CPU usage manageable. |
| `ROI_TOP` / `_BOTTOM` / `_LEFT` / `_RIGHT` | `0.25 / 0.92 / 0.05 / 0.95` | Fraction of the frame to crop before motion detection and classification. |
| `LOCAL_TIMEZONE` | `America/New_York` | Timezone used for sighting timestamps. Change this is you live elsewhere. |

---

## Project Structure (quick reference)

```
Bird-Nerd/
├── motion_camera/              # Production system (use this)
│   ├── main.py                 # Main detection + classification loop
│   ├── config.py               # All tunable settings
│   ├── bird_classify.py        # TFLite inference + multi-frame voting
│   ├── frame_capture.py        # Camera interface and burst recording
│   ├── motion_detect.py        # ROI crop and motion detection
│   ├── gif_builder.py          # Builds animated GIFs and thumbnails
│   ├── firebase_upload.py      # Uploads sightings to Firebase
│   ├── firebase_helper.py      # Sends test sightings to Firebase
│   ├── sighting_log.py         # Writes to local sightings.log
│   ├── setup_bird_model.py     # Downloads the TFLite model from Kaggle
│   ├── convert_to_tflite.py    # (Optional) convert SavedModel → TFLite
│   ├── models/                 # TFLite model + labels.txt (not committed)
│   ├── images/                 # High-confidence sighting images (not committed)
│   ├── unclear_images/         # Low-confidence images (not committed)
│   └── sightings.log           # Local text log (not committed)
└── website/                    # Static frontend
    ├── index.html
    ├── styles.css
    ├── firebase_functions.js
    ├── firebase_config.example.js  # Committed - placeholders only
    └── firebase_config.js          # Gitignored - your real keys go here
```
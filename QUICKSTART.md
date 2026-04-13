# Bird Nerd - Quick Start Guide (WIP)

> **Bird Nerd** is an IoT bird feeder system that uses motion detection and on-device TensorFlow Lite inference to automatically identify bird species, log sightings to Firebase, and display them on a live website.

---

## Prerequisites

- Raspberry Pi 4 (1 GB+ RAM)
- Raspberry Pi Camera Module (tested with `X000VGJ8BL`)
- MicroSD card (16 GB+ recommended) with Raspberry Pi OS (64-bit)
- Python 3.13
- A [Firebase](https://firebase.google.com/) account (free Spark tier is sufficient)

---

## 1. Hardware Setup

Attach the camera module to the Raspberry Pi's CSI camera port. That's it. Mount the Pi near your bird feeder with a clear line of sight. Make sure `rpicam-still` works before continuing:

```bash
rpicam-still -o test.jpg -t 2000
```

---

## 2. Clone the Repository

```bash
git clone https://github.com/BirdNerds/Bird-Nerd.git
cd Bird-Nerd
```

---

## 3. Set Up the Python Virtual Environment

```bash
cd motion_camera
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Note:** If `tflite-runtime` isn't available for your platform, install `tensorflow==2.20.0` instead - the code falls back automatically. A comment in `requirements.txt` explains this.

---

## 4. Download the Bird Classification Model

The project uses [Google's AIY Vision Classifier Birds V1](https://www.kaggle.com/models/google/aiy/tensorFlow1/vision-classifier-birds-v1) (964 species). A helper script handles the download.

You'll need a Kaggle account and API token first:

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

This downloads the model, copies it to `motion_camera/models/bird_classifier.tflite`, and downloads the species labels to `motion_camera/models/labels.txt`.

> If you already have a `.tflite` model file, just drop it into `motion_camera/models/` and name it `bird_classifier.tflite`. The labels file should be placed alongside it as `labels.txt`.

---

## 5. Firebase Setup

Bird Nerd uses Firebase for two things: **Firestore** (the sightings database) and **Firebase Storage** (storing bird photos).

### 5a. Create a Firebase Project

1. Go to the [Firebase Console](https://console.firebase.google.com/) and click **Add project**
2. Give it a name (e.g., `bird-nerd`) and follow the prompts
3. Once created, go to **Build → Firestore Database → Create database**
   - Choose **Start in production mode** (you'll add rules shortly)
   - Pick the region closest to you
4. Go to **Build → Storage → Get started** and follow the same steps

### 5b. Firestore Security Rules

In the Firebase Console under **Firestore → Rules**, paste:

```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /sightings/{sightingId} {
      allow read: if true;  // anyone can read
      allow delete: if request.auth != null && request.auth.token.email.matches('.*@birdnerd\\.local');
      allow create: if false;  // only your Python backend writes
    }
  }
}
```

This allows anyone to read sightings publicly, restricts deletes to authenticated admin users (the `@birdnerd.local` domain used by the website's login), and blocks direct creates from the browser entirely - only the Pi's service account can write new sightings. Click **Publish**.

Do the same for **Storage → Rules**:

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

### 5c. Generate a Service Account Key (for the Pi)

The Pi needs admin credentials to write sightings to Firebase.

1. In the Firebase Console, go to **Project Settings → Service accounts**
2. Click **Generate new private key** and download the JSON file
3. On the Pi, create the credentials directory and move the key there:

```bash
mkdir -p motion_camera/.credentials
mv ~/Downloads/your-key.json motion_camera/.credentials/bird-nerd-firebase-adminsdk.json
chmod 600 motion_camera/.credentials/bird-nerd-firebase-adminsdk.json
```

The JSON file Firebase generates will look roughly like this:

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

> ⚠️ **Never commit this file.** It is gitignored via `.credentials/` and `*-adminsdk*.json`. Do not modify the `.gitignore` in a way that would expose it. Treat it like a password.

### 5d. Create the `.env` File

A `.env.example` is included in `motion_camera/`. Copy it and fill in your values:

```bash
cd motion_camera
cp .env.example .env
```

Edit `motion_camera/.env`:

```
FIREBASE_CREDENTIALS_PATH=./.credentials/bird-nerd-firebase-adminsdk.json
FIREBASE_STORAGE_BUCKET=your-project-id.firebasestorage.app
```

Use a **relative path** for `FIREBASE_CREDENTIALS_PATH` (relative to `motion_camera/`) rather than an absolute path, so it works on any machine. Replace `your-project-id` with your actual Firebase project ID, visible in **Firebase Console → Project Settings**.

### 5e. Test the Firebase Connection

```bash
python3 firebase_helper.py
```

A successful run adds a test sighting to your Firestore database and prints the document ID. You can verify it appeared in the Firebase Console under **Firestore → sightings**.

---

## 6. Run the Detector

```bash
cd motion_camera
source venv/bin/activate
python3 main.py
```

The script will:
- Monitor for motion every 0.5 seconds
- Crop the region of interest and run the TFLite classifier when motion is detected
- Save images to `motion_camera/images/` (high confidence) or `motion_camera/unclear_images/` (low confidence)
- Log each sighting to `motion_camera/sightings.log` and upload it to Firebase

Stop it with `Ctrl+C`.

### Run on Boot (optional)

To have Bird Nerd start automatically when the Pi powers on, create a systemd service:

```bash
sudo nano /etc/systemd/system/birdnerd.service
```

```ini
[Unit]
Description=Bird Nerd Detection
After=network.target

[Service]
ExecStart=/home/pi/Bird-Nerd/motion_camera/venv/bin/python3 /home/pi/Bird-Nerd/motion_camera/main.py
WorkingDirectory=/home/pi/Bird-Nerd/motion_camera
User=pi
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable birdnerd
sudo systemctl start birdnerd
```

---

## 7. Website

The `website/` directory contains a static site (`index.html`, `styles.css`, `firebase_functions.js`) that reads from your Firestore database in real time.

### 7a. Firebase Config File

The website needs your Firebase project credentials, but these should never be committed to the repo. The config lives in a separate gitignored file, similar to how `.credentials/` works on the Pi side.

A `firebase_config.example.js` is included in `website/`. Copy it and fill in your values:

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
```

Find all of these values in the Firebase Console under **Project Settings → Your apps → Web app → SDK setup and configuration**. If you haven't registered a web app yet, click **Add app → Web** and follow the prompts.

> ⚠️ `firebase_config.js` is gitignored - never commit it. Only `firebase_config.example.js` (with placeholders) belongs in the repo.

### 7b. Running Locally

```bash
cd website
python3 -m http.server 8080
# Open http://localhost:8080 in your browser
```

The site reads directly from Firebase, so all your sightings will appear even when running locally - no Pi required.

### 7c. Hosting Publicly

There are many free or inexpensive options: GitHub Pages, Netlify, Vercel, Firebase Hosting, or a university web server. The site is plain static HTML/JS and works on any of them. Just make sure `firebase_config.js` is present on the server (deploy it manually - don't commit it).

---

## Key Configuration Options

All tunable parameters live at the top of `motion_camera/main.py`:

| Parameter | Default | Description |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.6` | Below this, images go to `unclear_images/` instead of `images/` |
| `COOLDOWN_PERIOD` | `10.0s` | Seconds to wait after a detection before checking again |
| `CHECK_INTERVAL` | `0.5s` | How often to capture a frame and check for motion |
| `MOTION_THRESHOLD` | `40` | Pixel difference threshold for motion detection |
| `MIN_AREA` | `2000` | Minimum contour area (px²) to count as motion |
| `LARGE_MOVE_RATIO` | `0.6` | Skip classification if moving region covers >60% of frame |
| `LAP_VAR_THRESHOLD` | `12.0` | Skip classification if crop is too smooth (catches blank sky, etc.) |
| `LOCAL_TIMEZONE` | `America/New_York` | Timezone used for log timestamps |

---

## Deprecated: Ollama Toy Model

The `toy_model/` directory contains an early prototype that used [Ollama's](https://ollama.com/) `llava:7b` vision model. You'd pass it an image and it would respond with a best-guess species in plain English. It worked, but at 30+ seconds per image it was far too slow and resource-intensive for the Raspberry Pi. It is no longer being developed and exists in the repo for historical reference only. If you want to experiment with it on more powerful hardware, see `toy_model/bird_nerd.py`.

---

## Project Structure (quick reference)

```
Bird-Nerd/
├── motion_camera/          # Production system (use this)
│   ├── main.py             # Main detection + classification loop
│   ├── firebase_helper.py  # Firestore & Storage upload
│   ├── setup_bird_model.py # Downloads the TFLite model
│   ├── convert_to_tflite.py# (Optional) convert SavedModel → TFLite
│   ├── models/             # TFLite model + labels.txt
│   ├── images/             # High-confidence sighting images
│   ├── unclear_images/     # Low-confidence images
│   └── sightings.log       # Local text log
├── website/                # Static frontend
│   ├── index.html
│   ├── styles.css
│   ├── firebase_functions.js
│   ├── firebase_config.example.js  # Committed - placeholders only
│   └── firebase_config.js          # Gitignored - your real keys go here
└── toy_model/              # Deprecated Ollama prototype, for fun
```

"""
firebase_upload.py - Upload sightings to Firestore and Firebase Storage.

Replaces firebase_helper.py with a cleaner interface that supports both
a JPEG thumbnail and a GIF URL in the same Firestore document.

Public API
----------
  initialize()                      -> bool
  upload_sighting(sighting: dict)   -> str | None   (Firestore document ID)

The sighting dict expected by upload_sighting():
  {
      "common_name":      str,
      "scientific_name":  str,
      "confidence":       float,        # 0.0 - 1.0
      "top_3_predictions": list,        # [(label, conf), ...]
      "thumb_path":       str | None,   # local JPEG path
      "gif_path":         str | None,   # local GIF path
      "timestamp":        datetime,
      "timezone":         str,          # e.g. "America/New_York"
  }

Firestore document schema (matches existing firebase_functions.js):
  timestamp           Firestore Timestamp
  timezone            str
  common_name         str
  scientific_name     str
  confidence          float
  top_3_predictions   [{label, confidence}, ...]
  image_url           str | null    ← JPEG thumbnail (was the only URL before)
  gif_url             str | null    ← NEW: animated GIF
"""

import os

import firebase_admin
from firebase_admin import credentials, firestore, storage
from dotenv import load_dotenv

load_dotenv()

_SERVICE_ACCOUNT = os.getenv(
    "FIREBASE_CREDENTIALS_PATH",
    "./.credentials/bird-nerd-firebase-adminsdk.json",
)
_STORAGE_BUCKET = os.getenv(
    "FIREBASE_STORAGE_BUCKET",
    "bird-nerd-27eb1.firebasestorage.app",
)

_initialized = False


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def initialize() -> bool:
    """
    Initialize the Firebase Admin SDK.
    Safe to call multiple times - subsequent calls are no-ops.
    Returns True on success, False on failure.
    """
    global _initialized
    if _initialized:
        return True
    if not os.path.exists(_SERVICE_ACCOUNT):
        return False
    try:
        cred = credentials.Certificate(_SERVICE_ACCOUNT)
        firebase_admin.initialize_app(cred, {"storageBucket": _STORAGE_BUCKET})
        _initialized = True
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------

def _upload_file(local_path: str, storage_path: str) -> str | None:
    """
    Upload a local file to Firebase Storage and return its public URL.
    Returns None on failure.
    """
    try:
        bucket = storage.bucket()
        blob   = bucket.blob(storage_path)
        blob.upload_from_filename(local_path)
        blob.make_public()
        return blob.public_url
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

def upload_sighting(sighting: dict) -> str | None:
    """
    Write one sighting to Firestore and upload its media files to Storage.

    Returns the Firestore document ID on success, None on failure.
    """
    if not _initialized and not initialize():
        return None

    db = firestore.client()

    # Format top_3 for Firestore (list of dicts)
    top3_formatted = [
        {"label": lbl, "confidence": float(conf)}
        for lbl, conf in sighting.get("top_3_predictions", [])
    ]

    doc_data = {
        "timestamp":         sighting["timestamp"],
        "timezone":          sighting.get("timezone", "America/New_York"),
        "common_name":       sighting["common_name"],
        "scientific_name":   sighting["scientific_name"],
        "confidence":        float(sighting["confidence"]),
        "top_3_predictions": top3_formatted,
        "image_url":         None,   # filled in below
        "gif_url":           None,   # filled in below
    }

    # Write document first to get the ID
    _, doc_ref = db.collection("sightings").add(doc_data)
    doc_id     = doc_ref.id

    updates: dict = {}

    # Upload JPEG thumbnail
    thumb = sighting.get("thumb_path")
    if thumb and os.path.exists(thumb):
        url = _upload_file(thumb, f"sightings/{doc_id}.jpg")
        if url:
            updates["image_url"] = url

    # Upload GIF
    gif = sighting.get("gif_path")
    if gif and os.path.exists(gif):
        url = _upload_file(gif, f"sightings/{doc_id}.gif")
        if url:
            updates["gif_url"] = url

    if updates:
        doc_ref.update(updates)

    return doc_id


def delete_sighting(doc_id: str) -> bool:
    """
    Delete a Firestore document and its associated Storage files.
    Returns True on success.
    """
    if not _initialized and not initialize():
        return False
    try:
        db     = firestore.client()
        bucket = storage.bucket()
        db.collection("sightings").document(doc_id).delete()
        for ext in ("jpg", "gif"):
            try:
                bucket.blob(f"sightings/{doc_id}.{ext}").delete()
            except Exception:
                pass   # file may not exist
        return True
    except Exception:
        return False

def send_heartbeat(alive: bool) -> None:
    """Write a heartbeat document to Firestore for the Pi status indicator."""
    try:
        db = firestore.client()
        db.collection('heartbeat').document('status').set({
            'alive':     alive,
            'timestamp': firestore.SERVER_TIMESTAMP,
        })
    except Exception as e:
        print(f"Heartbeat write failed: {e}")
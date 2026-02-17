"""
Firebase Helper Module
Handles uploading bird sightings to Firestore database
"""

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage
from datetime import datetime
import os
from dotenv import load_dotenv
from datetime import datetime, timezone

# Load environment variables from .env file
load_dotenv()

# Get credentials path from environment variable, with fallback
SERVICE_ACCOUNT_KEY = os.getenv('FIREBASE_CREDENTIALS_PATH', 
                                 '/home/tocila/Documents/Bird-Nerd/motion_camera/.credentials/bird-nerd-firebase-adminsdk.json')

# Firebase Storage bucket name (from your Firebase console — projectId.appspot.com)
STORAGE_BUCKET = os.getenv('FIREBASE_STORAGE_BUCKET', 'bird-nerd-27eb1.firebasestorage.app')

# Initialize Firebase (only once)
_firebase_initialized = False

def initialize_firebase():
    """Initialize Firebase Admin SDK"""
    global _firebase_initialized
    
    if _firebase_initialized:
        return True
    
    try:
        # Check if service account key exists
        if not os.path.exists(SERVICE_ACCOUNT_KEY):
            print(f"ERROR: Firebase service account key not found at {SERVICE_ACCOUNT_KEY}")
            print("Please check:")
            print("1. The .env file exists and has FIREBASE_CREDENTIALS_PATH set")
            print("2. The credentials JSON file is in the .credentials directory")
            return False
        
        # Initialize the app with service account
        cred = credentials.Certificate(SERVICE_ACCOUNT_KEY)
        firebase_admin.initialize_app(cred, {
            'storageBucket': STORAGE_BUCKET
        })
        
        _firebase_initialized = True
        print(f"Firebase initialized successfully")
        print(f"  Using credentials: {SERVICE_ACCOUNT_KEY}")
        return True
        
    except Exception as e:
        print(f"Error initializing Firebase: {e}")
        return False

def add_bird_sighting(common_name, scientific_name, confidence, top_3_predictions, timestamp=None, timezone=None, image_path=None):
    """
    Add a bird sighting to Firestore and optionally upload its image to Firebase Storage.
    
    Args:
        common_name (str): Common name of the bird (e.g., "Northern Cardinal")
        scientific_name (str): Scientific name (e.g., "Cardinalis cardinalis")
        confidence (float): Confidence score (0.0 to 1.0)
        top_3_predictions (list): List of tuples [(label, confidence), ...]
        timestamp (datetime, optional): Timestamp of sighting. Defaults to now.
        timezone (str, optional): Timezone string (e.g., "America/New_York"). Defaults to system timezone.
        image_path (str, optional): Local path to the image file to upload to Firebase Storage.
    
    Returns:
        str: Document ID if successful, None if failed
    """
    # Initialize Firebase if not already done
    if not _firebase_initialized:
        if not initialize_firebase():
            return None
    
    try:
        # Get Firestore client
        db = firestore.client()
        
        # Use provided timestamp or current time
        if timestamp is None:
            try:
                from tzlocal import get_localzone
                local_tz = get_localzone()
                timestamp = datetime.now(local_tz)
            except ImportError:
                # Fallback to UTC if tzlocal not available
                from datetime import timezone as dt_timezone
                timestamp = datetime.now(dt_timezone.utc)
        
        # Get timezone if not provided
        if timezone is None:
            import time
            if time.daylight:
                timezone_offset = -time.altzone
            else:
                timezone_offset = -time.timezone
            
            # Convert offset to hours
            tz_hours = timezone_offset // 3600
            tz_name = f"UTC{tz_hours:+d}"  # e.g., "UTC-5"
            
            # Try to get actual timezone name (requires tzlocal package)
            try:
                from tzlocal import get_localzone
                tz_name = str(get_localzone())  # e.g., "America/New_York"
            except ImportError:
                pass  # Use UTC offset format
            
            timezone = tz_name
        
        # Format top 3 predictions for storage
        top_3_formatted = [
            {
                'label': label,
                'confidence': float(conf)
            }
            for label, conf in top_3_predictions
        ]
        
        # Create document data (image_url filled in after upload below)
        sighting_data = {
            'timestamp': timestamp,
            'timezone': timezone,
            'common_name': common_name,
            'scientific_name': scientific_name,
            'confidence': float(confidence),
            'top_3_predictions': top_3_formatted,
            'image_url': None
        }
        
        # Add to Firestore first so we have the document ID
        doc_ref = db.collection('sightings').add(sighting_data)
        
        # doc_ref is a tuple: (timestamp, DocumentReference)
        doc_id = doc_ref[1].id
        
        print(f"Logged to Firebase: {common_name} ({confidence:.2%}) [ID: {doc_id[:8]}...] [{timezone}]")
        
        # -- Image upload ------------------------------------------------
        if image_path:
            image_url = upload_sighting_image(image_path, doc_id)
            if image_url:
                # Update the document with the public image URL
                doc_ref[1].update({'image_url': image_url})
                print(f"Image uploaded: sightings/{doc_id}.jpg")
            else:
                _log_image_warning(common_name, doc_id, image_path)
        
        return doc_id
        
    except Exception as e:
        print(f"Error adding bird sighting to Firebase: {e}")
        return None


def upload_sighting_image(image_path, doc_id):
    """
    Upload a local image file to Firebase Storage under sightings/{doc_id}.jpg
    Returns the public download URL on success, or None on failure.

    Args:
        image_path (str): Local path to the image file.
        doc_id (str): Firestore document ID used as the storage filename.

    Returns:
        str: Public download URL, or None if upload failed.
    """
    try:
        bucket = storage.bucket()
        blob_path = f"sightings/{doc_id}.jpg"
        blob = bucket.blob(blob_path)
        
        # Upload the file; JPEG content type is safe for both IMAGES_DIR and UNCLEAR_DIR
        blob.upload_from_filename(image_path, content_type='image/jpeg')
        
        # Make the file publicly readable so the browser can load it directly
        blob.make_public()
        
        return blob.public_url
    
    except Exception as e:
        print(f"Warning: Image upload failed for doc {doc_id}: {e}")
        return None


def _log_image_warning(common_name, doc_id, image_path):
    """
    Append a warning to sightings.log when an image upload fails.
    Matches the log format used by log_detection() in bird_ID_tflite.py.

    Args:
        common_name (str): Bird common name for context.
        doc_id (str): Firestore document ID of the affected sighting.
        image_path (str): Local path that failed to upload.
    """
    import time as _time
    from pathlib import Path as _Path
    
    # Best-effort: find the log file relative to this module, same as bird_ID_tflite.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, "sightings.log")
    
    warning_line = (
        f"[IMAGE UPLOAD WARNING] {_time.strftime('%Y-%m-%d %H:%M:%S')} "
        f"Failed to upload image for '{common_name}' (doc: {doc_id}). "
        f"Local path: {image_path}\n"
    )
    try:
        with open(log_file, 'a') as f:
            f.write(warning_line)
    except Exception as log_err:
        print(f"  Also failed to write warning to log: {log_err}")
    
    print(f"Warning: Image upload failed for '{common_name}' [doc: {doc_id[:8]}...] "
          f"- 'No image available' will be shown in the dashboard.")


def get_recent_sightings(limit=10):
    """
    Get recent bird sightings (for testing)
    
    Args:
        limit (int): Maximum number of sightings to retrieve
    
    Returns:
        list: List of sighting dictionaries
    """
    if not _firebase_initialized:
        if not initialize_firebase():
            return []
    
    try:
        db = firestore.client()
        
        # Query recent sightings, ordered by timestamp
        sightings_ref = db.collection('sightings').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit)
        
        sightings = []
        for doc in sightings_ref.stream():
            data = doc.to_dict()
            data['id'] = doc.id
            sightings.append(data)
        
        return sightings
        
    except Exception as e:
        print(f"Error retrieving sightings: {e}")
        return []

# Test function
if __name__ == "__main__":
    """Test the Firebase connection"""
    print("=" * 60)
    print("Firebase Connection Test")
    print("=" * 60)
    
    # Initialize
    if initialize_firebase():
        print("\nFirebase initialized successfully!")
        
        # Test adding a sighting
        print("\nAdding test sighting...")
        test_top_3 = [
            ("Cardinalis cardinalis (Northern Cardinal)", 0.9987),
            ("Spinus tristis (American Goldfinch)", 0.0008),
            ("Sialia sialis (Eastern Bluebird)", 0.0005)
        ]
        
        # Use the placeholder image bundled in this directory for test runs
        _placeholder_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_bird_image.png")
        _test_image = _placeholder_img if os.path.exists(_placeholder_img) else None
        if _test_image:
            print(f"  Using placeholder image: {_placeholder_img}")
        else:
            print(f"  Warning: fake_bird_image.png not found at {_placeholder_img} - uploading without image")

        doc_id = add_bird_sighting(
            common_name="Northern Cardinal",
            scientific_name="Cardinalis cardinalis",
            confidence=0.9987,
            top_3_predictions=test_top_3,
            image_path=_test_image
        )
        
        if doc_id:
            print(f"\nTest sighting added successfully!")
            print(f"  Document ID: {doc_id}")
            
            # Retrieve recent sightings
            print("\n" + "=" * 60)
            print("Retrieving recent sightings...")
            print("=" * 60)
            recent = get_recent_sightings(limit=5)
            print(f"\nFound {len(recent)} recent sightings:\n")
            for i, sighting in enumerate(recent, 1):
                timestamp = sighting['timestamp'].strftime("%Y-%m-%d %H:%M:%S") if hasattr(sighting['timestamp'], 'strftime') else str(sighting['timestamp'])
                print(f"{i}. {sighting['common_name']}")
                print(f"   Scientific: {sighting['scientific_name']}")
                print(f"   Confidence: {sighting['confidence']:.2%}")
                print(f"   Time: {timestamp}")
                print()
            
            print("=" * 60)
            print("All tests passed! Firebase is working correctly.")
            print("=" * 60)
        else:
            print("\nFailed to add test sighting")
    else:
        print("\nFirebase initialization failed")
        print("\n" + "=" * 60)
        print("Troubleshooting Checklist:")
        print("=" * 60)
        print("1. Check .env file exists with FIREBASE_CREDENTIALS_PATH")
        print("2. Verify credentials file exists in .credentials directory")
        print("3. Confirm file permissions (chmod 600 on JSON file)")
        print("4. Make sure python-dotenv is installed: pip install python-dotenv")
        print("5. Ensure firebase-admin is installed: pip install firebase-admin")
        print("=" * 60)
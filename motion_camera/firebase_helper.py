"""
Firebase Helper Module
Handles uploading bird sightings to Firestore database
"""

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from datetime import datetime
import os
from dotenv import load_dotenv
from datetime import datetime, timezone

# Load environment variables from .env file
load_dotenv()

# Get credentials path from environment variable, with fallback
SERVICE_ACCOUNT_KEY = os.getenv('FIREBASE_CREDENTIALS_PATH', 
                                 '/home/tocila/Documents/Bird-Nerd/motion_camera/.credentials/bird-nerd-firebase-adminsdk.json')

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
        firebase_admin.initialize_app(cred)
        
        _firebase_initialized = True
        print(f"✓ Firebase initialized successfully")
        print(f"  Using credentials: {SERVICE_ACCOUNT_KEY}")
        return True
        
    except Exception as e:
        print(f"Error initializing Firebase: {e}")
        return False

def add_bird_sighting(common_name, scientific_name, confidence, top_3_predictions, timestamp=None, timezone=None):
    """
    Add a bird sighting to Firestore
    
    Args:
        common_name (str): Common name of the bird (e.g., "Northern Cardinal")
        scientific_name (str): Scientific name (e.g., "Cardinalis cardinalis")
        confidence (float): Confidence score (0.0 to 1.0)
        top_3_predictions (list): List of tuples [(label, confidence), ...]
        timestamp (datetime, optional): Timestamp of sighting. Defaults to now.
        timezone (str, optional): Timezone string (e.g., "America/New_York"). Defaults to system timezone.
    
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
        
        # Create document data
        sighting_data = {
            'timestamp': timestamp,
            'timezone': timezone,  # NEW: Store timezone
            'common_name': common_name,
            'scientific_name': scientific_name,
            'confidence': float(confidence),
            'top_3_predictions': top_3_formatted,
            # We'll add image_url later when we implement image storage
            'image_url': None
        }
        
        # Add to Firestore collection
        doc_ref = db.collection('sightings').add(sighting_data)
        
        # doc_ref is a tuple: (timestamp, DocumentReference)
        doc_id = doc_ref[1].id
        
        print(f"✓ Logged to Firebase: {common_name} ({confidence:.2%}) [ID: {doc_id[:8]}...] [{timezone}]")
        return doc_id
        
    except Exception as e:
        print(f"Error adding bird sighting to Firebase: {e}")
        return None

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
        print("\n✓ Firebase initialized successfully!")
        
        # Test adding a sighting
        print("\nAdding test sighting...")
        test_top_3 = [
            ("Cardinalis cardinalis (Northern Cardinal)", 0.9987),
            ("Spinus tristis (American Goldfinch)", 0.0008),
            ("Sialia sialis (Eastern Bluebird)", 0.0005)
        ]
        
        doc_id = add_bird_sighting(
            common_name="Northern Cardinal",
            scientific_name="Cardinalis cardinalis",
            confidence=0.9987,
            top_3_predictions=test_top_3
        )
        
        if doc_id:
            print(f"\n✓ Test sighting added successfully!")
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
            print("✓ All tests passed! Firebase is working correctly.")
            print("=" * 60)
        else:
            print("\n✗ Failed to add test sighting")
    else:
        print("\n✗ Firebase initialization failed")
        print("\n" + "=" * 60)
        print("Troubleshooting Checklist:")
        print("=" * 60)
        print("1. Check .env file exists with FIREBASE_CREDENTIALS_PATH")
        print("2. Verify credentials file exists in .credentials directory")
        print("3. Confirm file permissions (chmod 600 on JSON file)")
        print("4. Make sure python-dotenv is installed: pip install python-dotenv")
        print("5. Ensure firebase-admin is installed: pip install firebase-admin")
        print("=" * 60)

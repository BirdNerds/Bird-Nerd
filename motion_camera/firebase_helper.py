"""
Firebase Helper Module
Handles uploading bird sightings to Firestore database
"""

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from datetime import datetime
import os

# Path to your service account key file
# UPDATE THIS PATH to where you saved the JSON file on your Pi
SERVICE_ACCOUNT_KEY = "/home/tocila/Documents/Bird-Nerd/motion_camera/bird-nerd-firebase-adminsdk.json"

# Initialize Firebase (only once)
_firebase_initialized = False

def initialize_firebase():
    """Initialize Firebase Admin SDK"""
    global _firebase_initialized
    
    if _firebase_initialized:
        return
    
    try:
        # Check if service account key exists
        if not os.path.exists(SERVICE_ACCOUNT_KEY):
            print(f"ERROR: Firebase service account key not found at {SERVICE_ACCOUNT_KEY}")
            print("Please update the SERVICE_ACCOUNT_KEY path in firebase_helper.py")
            return False
        
        # Initialize the app with service account
        cred = credentials.Certificate(SERVICE_ACCOUNT_KEY)
        firebase_admin.initialize_app(cred)
        
        _firebase_initialized = True
        print("Firebase initialized successfully")
        return True
        
    except Exception as e:
        print(f"Error initializing Firebase: {e}")
        return False

def add_bird_sighting(common_name, scientific_name, confidence, top_3_predictions, timestamp=None):
    """
    Add a bird sighting to Firestore
    
    Args:
        common_name (str): Common name of the bird (e.g., "Northern Cardinal")
        scientific_name (str): Scientific name (e.g., "Cardinalis cardinalis")
        confidence (float): Confidence score (0.0 to 1.0)
        top_3_predictions (list): List of tuples [(label, confidence), ...]
        timestamp (datetime, optional): Timestamp of sighting. Defaults to now.
    
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
            timestamp = datetime.now()
        
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
        
        print(f"✓ Bird sighting added to Firebase: {common_name} (ID: {doc_id})")
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
    print("Testing Firebase connection...")
    
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
            print(f"✓ Test sighting added successfully! Document ID: {doc_id}")
            
            # Retrieve recent sightings
            print("\nRetrieving recent sightings...")
            recent = get_recent_sightings(limit=5)
            print(f"Found {len(recent)} recent sightings:")
            for i, sighting in enumerate(recent, 1):
                print(f"  {i}. {sighting['common_name']} - {sighting['confidence']:.2%} - {sighting['timestamp']}")
        else:
            print("✗ Failed to add test sighting")
    else:
        print("\n✗ Firebase initialization failed")
        print("\nTroubleshooting:")
        print("1. Make sure you've downloaded the service account JSON file")
        print("2. Update SERVICE_ACCOUNT_KEY path at the top of this file")
        print("3. Run: pip install firebase-admin")

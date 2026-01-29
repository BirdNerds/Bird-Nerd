"""
Database Helper for Bird Detection System
Manages SQLite database for storing bird sightings and metadata
"""

import sqlite3
import os
from datetime import datetime
from pathlib import Path
import subprocess
from typing import Optional, List, Tuple, Dict

class BirdDatabase:
    """SQLite database manager for bird sightings"""
    
    def __init__(self, db_path: str = "birds.db"):
        """
        Initialize database connection
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Access columns by name
        self.cursor = self.conn.cursor()
    
    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        # Main sightings table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS sightings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                common_name TEXT NOT NULL,
                scientific_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                image_filename TEXT NOT NULL,
                image_path TEXT NOT NULL,
                uploaded_to_server BOOLEAN DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Table for top 3 predictions per sighting
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sighting_id INTEGER NOT NULL,
                rank INTEGER NOT NULL,
                species_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                FOREIGN KEY (sighting_id) REFERENCES sightings(id) ON DELETE CASCADE
            )
        """)
        
        # Index for faster queries
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON sightings(timestamp DESC)
        """)
        
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_uploaded 
            ON sightings(uploaded_to_server)
        """)
        
        self.conn.commit()
    
    def add_sighting(self, 
                     common_name: str,
                     scientific_name: str, 
                     confidence: float,
                     image_filename: str,
                     image_path: str,
                     top_3_predictions: List[Tuple[str, float]],
                     timestamp: Optional[datetime] = None) -> int:
        """
        Add a new bird sighting to the database
        
        Args:
            common_name: Common name of the bird
            scientific_name: Scientific name of the bird
            confidence: Confidence score (0-1)
            image_filename: Name of the image file
            image_path: Full path to the image file
            top_3_predictions: List of (species_name, confidence) tuples
            timestamp: Detection timestamp (defaults to now)
        
        Returns:
            sighting_id: ID of the inserted sighting
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Insert main sighting
        self.cursor.execute("""
            INSERT INTO sightings 
            (timestamp, common_name, scientific_name, confidence, 
             image_filename, image_path, uploaded_to_server)
            VALUES (?, ?, ?, ?, ?, ?, 0)
        """, (timestamp, common_name, scientific_name, confidence, 
              image_filename, image_path))
        
        sighting_id = self.cursor.lastrowid
        
        # Insert top 3 predictions
        for rank, (species_name, pred_confidence) in enumerate(top_3_predictions, 1):
            self.cursor.execute("""
                INSERT INTO predictions (sighting_id, rank, species_name, confidence)
                VALUES (?, ?, ?, ?)
            """, (sighting_id, rank, species_name, pred_confidence))
        
        self.conn.commit()
        return sighting_id
    
    def mark_as_uploaded(self, sighting_id: int):
        """Mark a sighting as uploaded to the server"""
        self.cursor.execute("""
            UPDATE sightings 
            SET uploaded_to_server = 1 
            WHERE id = ?
        """, (sighting_id,))
        self.conn.commit()
    
    def get_pending_uploads(self) -> List[Dict]:
        """
        Get all sightings that haven't been uploaded to the server
        
        Returns:
            List of sighting dictionaries
        """
        self.cursor.execute("""
            SELECT * FROM sightings 
            WHERE uploaded_to_server = 0
            ORDER BY timestamp ASC
        """)
        
        rows = self.cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_recent_sightings(self, limit: int = 10) -> List[Dict]:
        """
        Get the most recent bird sightings
        
        Args:
            limit: Maximum number of sightings to return
        
        Returns:
            List of sighting dictionaries with predictions
        """
        self.cursor.execute("""
            SELECT * FROM sightings 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,))
        
        sightings = [dict(row) for row in self.cursor.fetchall()]
        
        # Add predictions for each sighting
        for sighting in sightings:
            self.cursor.execute("""
                SELECT rank, species_name, confidence 
                FROM predictions 
                WHERE sighting_id = ?
                ORDER BY rank ASC
            """, (sighting['id'],))
            
            sighting['predictions'] = [dict(row) for row in self.cursor.fetchall()]
        
        return sightings
    
    def get_species_count(self) -> Dict[str, int]:
        """
        Get count of sightings per species
        
        Returns:
            Dictionary mapping scientific_name to count
        """
        self.cursor.execute("""
            SELECT scientific_name, COUNT(*) as count
            FROM sightings
            GROUP BY scientific_name
            ORDER BY count DESC
        """)
        
        return {row['scientific_name']: row['count'] for row in self.cursor.fetchall()}
    
    def get_total_sightings(self) -> int:
        """Get total number of sightings"""
        self.cursor.execute("SELECT COUNT(*) as count FROM sightings")
        return self.cursor.fetchone()['count']
    
    def get_sighting_by_id(self, sighting_id: int) -> Optional[Dict]:
        """
        Get a specific sighting by ID
        
        Args:
            sighting_id: ID of the sighting
        
        Returns:
            Sighting dictionary or None if not found
        """
        self.cursor.execute("""
            SELECT * FROM sightings WHERE id = ?
        """, (sighting_id,))
        
        row = self.cursor.fetchone()
        if row is None:
            return None
        
        sighting = dict(row)
        
        # Add predictions
        self.cursor.execute("""
            SELECT rank, species_name, confidence 
            FROM predictions 
            WHERE sighting_id = ?
            ORDER BY rank ASC
        """, (sighting_id,))
        
        sighting['predictions'] = [dict(row) for row in self.cursor.fetchall()]
        
        return sighting
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


class ServerSync:
    """Handle syncing data to Calvin University server"""
    
    def __init__(self, 
                 username: str,
                 server: str = "students.cs.calvin.edu",
                 remote_path: str = "~/public_html"):
        """
        Initialize server sync configuration
        
        Args:
            username: Calvin username (e.g., 'jt42')
            server: Server hostname
            remote_path: Remote directory path
        """
        self.username = username
        self.server = server
        self.remote_path = remote_path
        self.remote_host = f"{username}@{server}"
    
    def upload_image(self, local_image_path: str, remote_subdir: str = "images") -> bool:
        """
        Upload an image file to the server using SCP
        
        Args:
            local_image_path: Path to local image file
            remote_subdir: Subdirectory on server (default: 'images')
        
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(local_image_path):
            print(f"Error: Image file not found: {local_image_path}")
            return False
        
        filename = os.path.basename(local_image_path)
        remote_dest = f"{self.remote_host}:{self.remote_path}/{remote_subdir}/{filename}"
        
        try:
            # Ensure remote directory exists
            mkdir_cmd = f"ssh {self.remote_host} 'mkdir -p {self.remote_path}/{remote_subdir}'"
            subprocess.run(mkdir_cmd, shell=True, check=True, capture_output=True)
            
            # Upload file
            scp_cmd = ['scp', '-q', local_image_path, remote_dest]
            result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"  ✓ Uploaded: {filename}")
                return True
            else:
                print(f"  ✗ SCP failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"  ✗ Upload timeout for {filename}")
            return False
        except Exception as e:
            print(f"  ✗ Upload error: {e}")
            return False
    
    def upload_database(self, local_db_path: str) -> bool:
        """
        Upload the SQLite database to the server
        
        Args:
            local_db_path: Path to local database file
        
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(local_db_path):
            print(f"Error: Database file not found: {local_db_path}")
            return False
        
        filename = os.path.basename(local_db_path)
        remote_dest = f"{self.remote_host}:{self.remote_path}/{filename}"
        
        try:
            # Upload database
            scp_cmd = ['scp', '-q', local_db_path, remote_dest]
            result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"  ✓ Database synced: {filename}")
                return True
            else:
                print(f"  ✗ Database sync failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"  ✗ Database upload timeout")
            return False
        except Exception as e:
            print(f"  ✗ Database upload error: {e}")
            return False
    
    def sync_sighting(self, db: BirdDatabase, sighting_id: int) -> bool:
        """
        Sync a single sighting (image + database) to the server
        
        Args:
            db: BirdDatabase instance
            sighting_id: ID of the sighting to sync
        
        Returns:
            True if successful, False otherwise
        """
        sighting = db.get_sighting_by_id(sighting_id)
        if not sighting:
            print(f"Error: Sighting {sighting_id} not found")
            return False
        
        print(f"\nSyncing to {self.server}...")
        
        # Upload image
        image_success = self.upload_image(sighting['image_path'])
        
        # Upload database
        db_success = self.upload_database(db.db_path)
        
        # Mark as uploaded if both succeeded
        if image_success and db_success:
            db.mark_as_uploaded(sighting_id)
            print("✓ Sync complete\n")
            return True
        else:
            print("✗ Sync failed\n")
            return False
    
    def test_connection(self) -> bool:
        """
        Test SSH connection to the server
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            test_cmd = f"ssh -o ConnectTimeout=5 {self.remote_host} 'echo Connection OK'"
            result = subprocess.run(test_cmd, shell=True, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and "Connection OK" in result.stdout:
                print(f"✓ Connection to {self.server} successful")
                return True
            else:
                print(f"✗ Connection failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"✗ Connection test error: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Example: Create database and add a sighting
    with BirdDatabase("birds.db") as db:
        # Add a test sighting
        sighting_id = db.add_sighting(
            common_name="Northern Cardinal",
            scientific_name="Cardinalis cardinalis",
            confidence=0.98,
            image_filename="2025_01_28_14_30_45_Northern_Cardinal.jpg",
            image_path="/home/claude/images/2025_01_28_14_30_45_Northern_Cardinal.jpg",
            top_3_predictions=[
                ("Cardinalis cardinalis (Northern Cardinal)", 0.98),
                ("Piranga olivacea (Scarlet Tanager)", 0.01),
                ("Passerina ciris (Painted Bunting)", 0.01)
            ]
        )
        
        print(f"Added sighting ID: {sighting_id}")
        
        # Get recent sightings
        recent = db.get_recent_sightings(limit=5)
        print(f"\nRecent sightings: {len(recent)}")
        
        # Get species counts
        counts = db.get_species_count()
        print(f"\nSpecies counts: {counts}")
    
    # Example: Test server sync (you'll need to update username)
    # sync = ServerSync(username="jt42")
    # if sync.test_connection():
    #     with BirdDatabase("birds.db") as db:
    #         sync.sync_sighting(db, sighting_id)

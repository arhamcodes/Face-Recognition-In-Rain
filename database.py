import sqlite3
from datetime import datetime
import os
from PIL import Image
from pathlib import Path
from passlib.hash import pbkdf2_sha256

DB_PATH = "data/app.db"
LOGS_PATH = Path("data/logs")
LOGS_PATH.mkdir(parents=True, exist_ok=True)

def init_db():
    """Initialize database with required tables"""
    LOGS_PATH.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Create users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT, role TEXT)''')
    
    # Create recognition logs table with proper indices
    c.execute('''CREATE TABLE IF NOT EXISTS recognition_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  person_name TEXT,
                  confidence REAL,
                  augmented_path TEXT,
                  derained_path TEXT,
                  UNIQUE(timestamp, person_name))''')
                  
    # Create index for faster lookups
    c.execute('''CREATE INDEX IF NOT EXISTS idx_timestamp 
                 ON recognition_logs(timestamp)''')
    
    # Create default admin user
    admin_pwd = pbkdf2_sha256.hash('admin123')
    c.execute(
        'INSERT OR IGNORE INTO users (username, password, role) VALUES (?, ?, ?)',
        ('admin', admin_pwd, 'admin')
    )
    
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully")

def get_db_connection():
    return sqlite3.connect(DB_PATH)

def verify_user(username, password):
    """Verify user credentials and return user data"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT password, role FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()
        
        if result and pbkdf2_sha256.verify(password, result[0]):
            return {
                'username': username,
                'role': result[1]
            }
        return None
    finally:
        conn.close()

def create_user(username: str, password: str, role: str = "user") -> bool:
    """Create a new app user"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        hash = pbkdf2_sha256.hash(password)  # Use pbkdf2 instead of sha256
        cursor.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            (username, hash, role)
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"Error creating user: {e}")
        return False
    finally:
        conn.close()

def get_all_users():
    """Get all app users"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT username, role FROM users")
        users = [{"username": row[0], "role": row[1]} for row in cursor.fetchall()]
        return users
    except Exception as e:
        print(f"Error getting users: {e}")
        return []
    finally:
        conn.close()

def delete_user(username: str) -> bool:
    """Delete an app user"""
    if username == "admin":  # Prevent deleting admin
        return False
        
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error deleting user: {e}")
        return False
    finally:
        conn.close()

def save_recognition(timestamp, name, confidence, original_img=None, derained_img=None):
    """Save recognition log with images"""
    # Ensure logs directory exists
    # LOGS_PATH = Path("data/logs")
    # LOGS_PATH.mkdir(parents=True, exist_ok=True)
    
    # Create processed_images directory
    PROCESSED_IMAGES_PATH = Path("data/processed_images")
    PROCESSED_IMAGES_PATH.mkdir(parents=True, exist_ok=True)

    # conn = sqlite3.connect(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        # Save images if provided
        augmented_path = None
        derained_path = None
        
        if original_img and derained_img:
            # Generate unique filenames based on timestamp
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            augmented_path = str(PROCESSED_IMAGES_PATH / f"augmented_{ts}.png")
            derained_path = str(PROCESSED_IMAGES_PATH / f"derained_{ts}.png")
            
            # Ensure images are in correct format
            if isinstance(original_img, Image.Image):
                original_img.convert('RGB').save(augmented_path, 'PNG')
            if isinstance(derained_img, Image.Image):
                derained_img.convert('RGB').save(derained_path, 'PNG')
            
            print(f"✅ Saved recognition images to {PROCESSED_IMAGES_PATH}")
        
        # Insert log entry
        c.execute('''INSERT INTO recognition_logs 
                     (timestamp, person_name, confidence, augmented_path, derained_path)
                     VALUES (?, ?, ?, ?, ?)''',
                  (timestamp, name, confidence, augmented_path, derained_path))
        
        log_id = c.lastrowid
        conn.commit()
        print(f"✅ Saved recognition log #{log_id} to database")
        return log_id
        
    except Exception as e:
        print(f"Error saving recognition log: {e}")
        conn.rollback()
        return None
    finally:
        conn.close()

def get_recognition(log_id):
    """Get recognition log by ID"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        c.execute('''SELECT * FROM recognition_logs WHERE id = ?''', (log_id,))
        row = c.fetchone()
        
        if row:
            return {
                'id': row[0],
                'timestamp': row[1],
                'person_name': row[2],
                'confidence': row[3],
                'augmented_path': row[4],
                'derained_path': row[5]
            }
        return None
    finally:
        conn.close()

def get_all_logs():
    """Get all recognition logs ordered by timestamp"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        c.execute('''SELECT * FROM recognition_logs 
                     ORDER BY datetime(timestamp) DESC''')
        
        logs = []
        for row in c.fetchall():
            logs.append({
                'id': row[0],
                'timestamp': row[1],
                'person_name': row[2],
                'confidence': row[3],
                'augmented_path': row[4],
                'derained_path': row[5]
            })
        return logs
    finally:
        conn.close()

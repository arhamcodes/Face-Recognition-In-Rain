import cv2
import numpy as np
import requests
import time
from PIL import Image
from io import BytesIO
import os

# Backend server URL - Change to your PC's IP address
BACKEND_URL = "http://192.168.1.x:8000"  # Replace with your PC's IP

def init_camera():
    """Initialize USB webcam"""
    cap = cv2.VideoCapture(0)  # Try 0 first, if not working try 1,2 etc.
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera. Check if it's connected properly.")
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("✅ Camera initialized successfully")
    return cap

def init_face_detector():
    """Initialize OpenCV face detector"""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    return face_cascade

def detect_faces(frame, face_cascade):
    """Detect faces in frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return len(faces) > 0, faces

def send_frame(frame):
    """Send frame to backend server"""
    try:
        # Convert frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        # Convert to bytes
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Send to backend
        files = {'file': ('frame.png', img_byte_arr, 'image/png')}
        response = requests.post(f"{BACKEND_URL}/process_image", files=files)
        
        if response.status_code == 200:
            print("✅ Frame processed successfully")
        else:
            print(f"❌ Failed to process frame: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error sending frame: {e}")

def main():
    print("Initializing Camera System...")
    
    # Initialize camera and face detector
    cap = init_camera()
    face_cascade = init_face_detector()
    
    print("✅ Camera system ready")
    print("Press Ctrl+C to exit")
    
    last_sent_time = 0
    min_send_interval = 0.5  # Minimum time between frame sends (seconds)
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                continue
            
            # Detect faces
            faces_detected, faces = detect_faces(frame, face_cascade)
            
            current_time = time.time()
            if faces_detected and (current_time - last_sent_time) >= min_send_interval:
                # Draw rectangles for visual feedback
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Save frame locally when face detected
                timestamp = int(time.time() * 1000)
                save_path = os.path.join('captured_frames', f'frame_{timestamp}.jpg')
                os.makedirs('captured_frames', exist_ok=True)
                cv2.imwrite(save_path, frame)
                print(f"✅ Saved frame: {save_path}")
                
                last_sent_time = current_time
            
            # Display frame locally
            cv2.imshow('Camera Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cv2.destroyAllWindows()
        cap.release()

if __name__ == "__main__":
    main()

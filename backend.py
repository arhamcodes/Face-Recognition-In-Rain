import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends, status
from fastapi.responses import Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from PIL import Image
import io
import numpy as np
from torchvision import transforms
import uvicorn
import os
from pathlib import Path
import time
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct, PointIdsList
from database import init_db, verify_user, create_user, get_all_users, delete_user
import queue
from datetime import datetime
from starlette.responses import StreamingResponse
import asyncio
import cv2

app = FastAPI()
security = HTTPBasic()

# Create directories for storing embeddings
EMBEDDINGS_DIR = Path("data/embeddings")
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

FRAMES_DIR = Path("data/frames")
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
MAX_STORED_FRAMES = 100

frames_queue = queue.Queue(maxsize=30)

class TripletNetwork(nn.Module):
    def __init__(self, device):
        super(TripletNetwork, self).__init__()
        self.resnet = InceptionResnetV1(pretrained='vggface2').to(device)

    def forward_once(self, x):
        return self.resnet(x)

    def forward(self, anchor, positive, negative):
        anchor_output = self.forward_once(anchor)
        positive_output = self.forward_once(positive)
        negative_output = self.forward_once(negative)
        return anchor_output, positive_output, negative_output

# Initialize Qdrant client and create collection
def init_qdrant():
    try:
        client = QdrantClient(
            url="http://localhost:6333",
            timeout=10.0
        )
        
        # Test connection
        client.get_collections()
        
        # Create collection only if it doesn't exist
        collections = client.get_collections().collections
        if not any(c.name == "faces" for c in collections):
            client.create_collection(
                collection_name="faces",
                vectors_config=VectorParams(size=512, distance=Distance.COSINE)
            )
            print("✅ Created new Qdrant collection 'faces'")
        else:
            print("✅ Using existing Qdrant collection 'faces'")
        return client
    except Exception as e:
        print(f"⚠️ Warning: Could not connect to Qdrant Docker container: {str(e)}")
        print("Make sure Qdrant Docker container is running on port 6333")
        return None

# Initialize components
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = torch.jit.load("models/generator_model.pt", map_location=device)
generator.eval()

face_model = torch.load("models/triplet_model_full.pt", map_location=device, weights_only=False)
face_model.eval()

@app.on_event("startup")
async def startup_event():
    """Initialize components on server startup"""
    print("Initializing server components...")
    # Initialize database
    init_db()
    # Initialize Qdrant
    global qdrant_client
    qdrant_client = init_qdrant()
    print("✅ Server initialization complete")

# Image Preprocessing Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def get_face_embedding(image):
    """Generate embedding for a face image"""
    try:
        # Ensure image is the right size and format
        image = transforms.Resize((256, 256))(image)  # FaceNet expects 
        image = transforms.ToTensor()(image)
        image = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(image)
        
        with torch.no_grad():
            image = image.unsqueeze(0).to(device)
            embedding = face_model.forward_once(image)
            return embedding.cpu().numpy()[0]
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

@app.post("/process_image")
async def process_image(file: UploadFile):
    try:
        # Read image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        
        # Process image with derain model
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = generator(img_tensor)
            output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        processed_frame = (output * 255).astype("uint8")
        
        # Convert back to image
        output_img = Image.fromarray(processed_frame)
        img_byte_arr = io.BytesIO()
        output_img.save(img_byte_arr, format='PNG')
        
        return Response(
            content=img_byte_arr.getvalue(),
            media_type="image/png",
            headers={"X-Processing-Time": str(time.time())}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/register_face")
async def register_face(
    file: UploadFile, 
    name: str = Form(...),
    user_id: str = Form(...)
):
    if qdrant_client is None:
        raise HTTPException(
            status_code=503, 
            detail="Face registration service unavailable - Qdrant server not connected"
        )
    try:
        # Check if user_id already exists
        existing_users = await get_registered_users()
        if any(user['user_id'] == user_id for user in existing_users['users']):
            raise HTTPException(status_code=400, detail=f"ID {user_id} is already registered")

        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Get embedding
        embedding = get_face_embedding(img)
        if embedding is None:
            raise HTTPException(status_code=400, detail="Could not generate face embedding")
        
        # Generate unique point ID
        point_id = abs(hash(f"{user_id}_{time.time()}"))
        
        # Store in Qdrant with user_id
        qdrant_client.upsert(
            collection_name="faces",
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "name": name,
                        "user_id": user_id,
                        "timestamp": time.time(),
                        "face_id": str(point_id)
                    }
                )
            ]
        )
        
        print(f"✅ Registered face for {name} (ID: {user_id})")
        return {
            "status": "success", 
            "message": f"Face registered for {name}",
            "face_id": str(point_id),
            "user_id": user_id
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error in registration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recognize_face")
async def recognize_face(file: UploadFile):
    if qdrant_client is None:
        raise HTTPException(
            status_code=503, 
            detail="Face recognition service unavailable - Qdrant server not connected"
        )
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        
        # Get embedding
        embedding = get_face_embedding(img)
        
        # Search in Qdrant
        search_result = qdrant_client.search(
            collection_name="faces",
            query_vector=embedding.tolist(),
            limit=1,
            score_threshold=0.8
        )
        
        
        
        if search_result:
            match = search_result[0]
            return {
                "name": match.payload["name"],
                "confidence": float(match.score)
            }
        return {"name": "Unknown", "confidence": 0.0}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/registered_users")
async def get_registered_users():
    if qdrant_client is None:
        raise HTTPException(status_code=503, detail="Qdrant server not connected")
    try:
        # Scroll through all points in the collection
        registered_users = []
        offset = 0
        limit = 100
        
        while True:
            points = qdrant_client.scroll(
                collection_name="faces",
                offset=offset,
                limit=limit
            )[0]
            
            if not points:
                break
                
            for point in points:
                registered_users.append({
                    "name": point.payload["name"],
                    "user_id": point.payload.get("user_id", None),
                    "face_id": point.payload["face_id"],
                    "timestamp": point.payload["timestamp"]
                })
            
            offset += limit
            if len(points) < limit:
                break
        
        return {"users": registered_users}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete_face/{face_id}")
async def delete_face(face_id: str):
    if qdrant_client is None:
        raise HTTPException(status_code=503, detail="Qdrant server not connected")
    try:
        # Delete the point from collection
        qdrant_client.delete(
            collection_name="faces",
            points_selector=PointIdsList(points=[int(face_id)])
        )
        return {"status": "success", "message": f"Face {face_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": generator is not None}

@app.post("/receive_frame")
async def receive_frame(file: UploadFile):
    """Receive frame from Raspberry Pi"""
    try:
        contents = await file.read()
        
        # Store frame in queue for live display
        try:
            frames_queue.put_nowait(contents)
        except queue.Full:
            try:
                frames_queue.get_nowait()
                frames_queue.put_nowait(contents)
            except queue.Empty:
                pass
        
        # Store frame on disk with timestamp
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            frame_path = FRAMES_DIR / f"frame_{timestamp}.png"
            
            # Remove old frames if too many
            stored_frames = list(FRAMES_DIR.glob("frame_*.png"))
            if len(stored_frames) > MAX_STORED_FRAMES:
                oldest_frames = sorted(stored_frames)[:-MAX_STORED_FRAMES]
                for old_frame in oldest_frames:
                    old_frame.unlink()
            
            # Save new frame
            with open(frame_path, "wb") as f:
                f.write(contents)
                
        except Exception as e:
            print(f"Error saving frame: {e}")
            
        return {"status": "success"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_latest_frame")
async def get_latest_frame():
    """Get latest frame for frontend"""
    try:
        frame_data = frames_queue.get_nowait()
        return Response(content=frame_data, media_type="image/png")
    except queue.Empty:
        raise HTTPException(status_code=404, detail="No frame available")

@app.get("/video_feed")
async def video_feed():
    """Video streaming endpoint"""
    async def generate():
        while True:
            try:
                frame_data = frames_queue.get_nowait()
                if frame_data:
                    # Convert PNG to JPEG for MJPEG streaming
                    frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
                    _, jpeg_frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    
                    yield b'--frame\r\n'
                    yield b'Content-Type: image/jpeg\r\n\r\n'
                    yield jpeg_frame.tobytes()
                    yield b'\r\n'
            except queue.Empty:
                await asyncio.sleep(0.03)
            except Exception as e:
                print(f"Video feed error: {e}")
                await asyncio.sleep(0.1)

    return StreamingResponse(
        generate(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    user = verify_user(credentials.username, credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return user

@app.post("/login")
async def login(credentials: HTTPBasicCredentials = Depends(security)):
    try:
        user = verify_user(credentials.username, credentials.password)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return user
    except Exception as e:
        print(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/create_user")
async def create_app_user(
    user_data: dict,
    current_user: dict = Depends(get_current_user)
):
    # Check if user has admin role
    if not current_user or current_user.get("role") != "admin":
        raise HTTPException(
            status_code=403,
            detail="Only admins can create users"
        )
    
    # Validate required fields
    if not all(k in user_data for k in ["username", "password", "role"]):
        raise HTTPException(
            status_code=400,
            detail="Missing required fields"
        )
    
    # Validate role
    if user_data["role"] not in ["admin", "user"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid role specified"
        )
        
    try:
        success = create_user(
            user_data["username"],
            user_data["password"],
            user_data["role"]
        )
        
        if success:
            return {"message": "User created successfully"}
        raise HTTPException(status_code=400, detail="Failed to create user")
    except Exception as e:
        print(f"Error creating user: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/users")
async def get_users(current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=403,
            detail="Only admins can view users"
        )
    return {"users": get_all_users()}

@app.delete("/users/{username}")
async def delete_app_user(
    username: str,
    current_user: dict = Depends(get_current_user)
):
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=403,
            detail="Only admins can delete users"
        )
    
    if username == "admin":
        raise HTTPException(
            status_code=400,
            detail="Cannot delete admin user"
        )
    
    success = delete_user(username)
    if success:
        return {"message": "User deleted successfully"}
    raise HTTPException(status_code=400, detail="Failed to delete user")

if __name__ == "__main__":
    print(f"Starting server... GPU available: {torch.cuda.is_available()}")
    uvicorn.run(app, host="0.0.0.0", port=8000)

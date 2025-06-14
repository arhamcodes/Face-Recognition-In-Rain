# Face Recognition System with Rain Removal

This project is a face recognition system with rain removal capabilities that uses a client-server architecture with a Raspberry Pi 5 camera client.

![Dataset](dataset.png)

## Prerequisites

- Docker installed on the main system
- Python 3.8+ installed on both main system and Raspberry Pi
- Raspberry Pi 5 with camera module
- Network connectivity between Pi and main system
- Generator and Face Recognition models can be downloaded from the drive folder and add to the "model" folder (https://drive.google.com/drive/folders/16XAeap0ioKyXb1oItKE83y9ugTFxd5eV?usp=sharing)

## System Architecture

- **Backend Server**: Handles face recognition and rain removal processing
- **Frontend**: GUI application for user interaction
- **Pi Camera**: Captures and streams video from Raspberry Pi

## Installation & Setup

### 1. Start Qdrant Vector Database

```bash
# Pull and run Qdrant container
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 2. Install Dependencies

Create a virtual environment and install dependencies on both main system and Pi:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/Pi:
source venv/bin/activate

# Install dependencies using poetry
pip install poetry
poetry install
```

### 3. Start Backend Server

```bash
# From project root directory
python backend.py
```

The backend will start on http://localhost:8000

### 4. Start Frontend Application

```bash
# From project root directory
python frontend.py
```

### 5. Configure and Start Pi Camera Client

1. Edit `pi_camera.py` on the Raspberry Pi:
   - Update `BACKEND_URL` to match your main system's IP address:
   ```python
   BACKEND_URL = "http://YOUR_PC_IP:8000"
   ```

2. Start the camera client:
```bash
# On Raspberry Pi
python pi_camera.py
```

## Default Login

- Username: admin
- Password: admin123
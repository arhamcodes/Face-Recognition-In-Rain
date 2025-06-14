import subprocess
import sys
import os
import signal
import atexit
import time
import psutil
import threading
import socket
import sqlite3
import requests
from colorama import init, deinit
import win32event
import win32api
import win32con  # Add this import
import winerror
import win32process
import logging
from datetime import datetime
from pathlib import Path

# Add mutex to prevent multiple instances
mutex_name = "Global\\FaceRecognitionSystem"
mutex = win32event.CreateMutex(None, 1, mutex_name)
if win32api.GetLastError() == winerror.ERROR_ALREADY_EXISTS:
    print("Application is already running")
    sys.exit(0)

# Setup logging
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)
log_file = LOGS_DIR / f"launcher_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def get_app_path():
    """Get base path for the application"""
    if getattr(sys, 'frozen', False):
        # Running as exe
        return os.path.dirname(sys.executable)
    # Running as script
    return os.path.dirname(os.path.abspath(__file__))

# Set working directory to exe/script location
os.chdir(get_app_path())

def get_python_executable():
    """Get correct Python executable path for subprocess"""
    if getattr(sys, 'frozen', False):
        # If running as exe, use the packaged Python
        return sys.executable
    else:
        # If running as script, use system Python
        return sys.executable

def kill_proc_tree(pid):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        try:
            parent.kill()
        except psutil.NoSuchProcess:
            pass
    except psutil.NoSuchProcess:
        pass

def monitor_process(process, timeout=20):
    start_time = time.time()
    while time.time() - start_time < timeout:
        if process.poll() is not None:  # Process ended
            return False
        time.sleep(0.1)
    
    # If we get here, process didn't end within timeout
    kill_proc_tree(process.pid)
    return True

def check_port_available(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('localhost', port))
        result = True
    except:
        result = False
    sock.close()
    return result

def check_port_in_use(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('localhost', port))
        sock.close()
        return False
    except socket.error:
        sock.close()
        return True

def wait_for_port(port, timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect(('localhost', port))
            sock.close()
            return True
        except:
            sock.close()
            time.sleep(1)
    return False

def check_database():
    if not os.path.exists("data"):
        os.makedirs("data")
    try:
        conn = sqlite3.connect("data/app.db")
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        conn.close()
        return len(tables) > 0
    except:
        return False

def extract_resources():
    """Extract required files when running as exe"""
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
        required_files = ['backend.py', 'frontend.py', 'database.py']
        for file in required_files:
            if not os.path.exists(os.path.join(exe_dir, file)):
                logging.error(f"Missing required file: {file}")
                return False
        return True
    return True

def main():
    logging.info("Starting application...")
    init()
    
    try:
        # Check resources first
        if not extract_resources():
            logging.error("Missing required files")
            sys.exit(1)

        # Set working directory
        app_path = get_app_path()
        os.chdir(app_path)
        logging.info(f"Working directory: {app_path}")

        # Kill existing processes
        logging.info("Checking for existing processes...")
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if 'python' in proc.info['name'].lower():
                    kill_proc_tree(proc.info['pid'])
                    time.sleep(1)
            except:
                continue

        # Start backend with proper path resolution
        logging.info("Starting backend server...")
        backend_path = os.path.join(app_path, "backend.py")
        
        if not os.path.exists(backend_path):
            logging.error(f"Backend not found at: {backend_path}")
            sys.exit(1)

        # Create startup info to properly handle console
        startupinfo = None
        if hasattr(subprocess, 'STARTUPINFO'):
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = win32con.SW_SHOW  # Use win32con instead

        backend_process = subprocess.Popen(
            [sys.executable, backend_path],
            env={
                **os.environ,
                'PYTHONPATH': app_path,
                'PYTHONUNBUFFERED': '1'
            },
            cwd=app_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=startupinfo,
            creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NO_WINDOW
        )

        # Monitor backend with better error handling
        logging.info("Waiting for backend initialization...")
        start_time = time.time()
        dots = 0
        
        while time.time() - start_time < 30:
            returncode = backend_process.poll()
            if returncode is not None:
                stdout, stderr = backend_process.communicate()
                logging.error(f"Backend failed to start:")
                logging.error(f"Exit code: {returncode}")
                logging.error(f"Stdout: {stdout.decode() if stdout else 'None'}")
                logging.error(f"Stderr: {stderr.decode() if stderr else 'None'}")
                sys.exit(1)

            try:
                response = requests.get('http://localhost:8000/health', timeout=1)
                if response.status_code == 200:
                    logging.info("\nBackend ready!")
                    break
            except requests.exceptions.RequestException:
                dots = (dots + 1) % 4
                print('.' * dots + '\r', end='', flush=True)
                time.sleep(1)
                continue
            except Exception as e:
                logging.error(f"Unexpected error checking backend: {e}")
                
        else:
            logging.error("\nBackend failed to respond in time")
            kill_proc_tree(backend_process.pid)
            sys.exit(1)

        # Start frontend
        logging.info("Starting frontend...")
        frontend_path = os.path.join(app_path, "frontend.py")
        
        if not os.path.exists(frontend_path):
            logging.error(f"Frontend not found at: {frontend_path}")
            sys.exit(1)

        frontend_process = subprocess.Popen(
            [sys.executable, frontend_path],
            env={
                **os.environ,
                'PYTHONPATH': app_path,
                'PYTHONUNBUFFERED': '1'
            },
            cwd=app_path,
            startupinfo=startupinfo,
            creationflags=subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NO_WINDOW
        )

        def cleanup():
            logging.info("Cleaning up processes...")
            kill_proc_tree(frontend_process.pid)
            time.sleep(0.5)
            kill_proc_tree(backend_process.pid)
            deinit()

        atexit.register(cleanup)

        # Monitor both processes with better error handling
        while True:
            backend_status = backend_process.poll()
            frontend_status = frontend_process.poll()

            if frontend_status is not None:
                logging.info(f"Frontend exited with code {frontend_status}")
                backend_process.kill()
                break

            if backend_status is not None:
                stdout, stderr = backend_process.communicate()
                logging.error(f"Backend crashed with code {backend_status}:")
                logging.error(f"Stdout: {stdout.decode() if stdout else 'None'}")
                logging.error(f"Stderr: {stderr.decode() if stderr else 'None'}")
                frontend_process.kill()
                sys.exit(1)

            time.sleep(0.1)

    except Exception as e:
        logging.exception("Critical error occurred")
        try:
            if 'backend_process' in locals():
                backend_process.kill()
            if 'frontend_process' in locals():
                frontend_process.kill()
        except:
            pass
        deinit()
        sys.exit(1)
    finally:
        # Release mutex on exit
        if mutex:
            win32api.CloseHandle(mutex)

if __name__ == "__main__":
    main()
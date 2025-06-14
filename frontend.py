import base64
import numpy as np
import cv2
import customtkinter as ctk
from tkinter import filedialog  # Add this import
import threading
import os
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import time
import requests
import json
from io import BytesIO
from datetime import datetime
from database import init_db, save_recognition, get_recognition, get_all_logs
from rain_utils import generate_rainy_image
import queue
import torch

# Add icons path constant
ICONS_PATH = Path("assets/icons")

# Backend server URL
BACKEND_URL = "http://localhost:8000"  # Change to your cloud server URL

def init_assets():
    """Initialize required directories and assets"""
    if not ICONS_PATH.exists():
        ICONS_PATH.mkdir(parents=True, exist_ok=True)

    required_icons = ['upload.png','face1.png','face2.png', 'add.png', 'chart.png', 'settings.png']
    missing_icons = []
    for icon in required_icons:
        if not (ICONS_PATH / icon).exists():
            missing_icons.append(icon)
    
    if missing_icons:
        print(f"Warning: Missing icons: {', '.join(missing_icons)}")
        print(f"Please add icons to: {ICONS_PATH}")

class LoadingScreen:
    def __init__(self, root):
        self.root = root
        self.running = True
        self.frame = ctk.CTkFrame(
            root,
            fg_color="#ffffff",
            width=400,
            height=200
        )
        self.frame.place(relx=0.5, rely=0.5, anchor="center")
        
        # Loading text
        self.loading_label = ctk.CTkLabel(
            self.frame,
            text="Initializing System...",
            font=("SF Pro Display", 24, "bold"),
            text_color="#000000"
        )
        self.loading_label.pack(pady=20)
        
        # Loading animation dots
        self.dot_label = ctk.CTkLabel(
            self.frame,
            text="",
            font=("SF Pro Display", 24, "bold"),
            text_color="#8ac4ff"
        )
        self.dot_label.pack(pady=10)
        
        self.dot_count = 0
        self.update_dots()
        
    def update_dots(self):
        if not self.running:
            return
        try:
            if self.dot_label is None:
                return
            dots = "." * ((self.dot_count % 3) + 1)
            self.dot_label.configure(text=dots)
            self.dot_count += 1
            if self.running:
                self.root.after(500, self.update_dots)
        except Exception:
            self.running = False
            
    def destroy(self):
        self.running = False
        try:
            if hasattr(self, 'dot_label'):
                self.dot_label.destroy()
                self.dot_label = None
            if hasattr(self, 'loading_label'):
                self.loading_label.destroy()
            if hasattr(self, 'frame'):
                self.frame.destroy()
        except Exception:
            pass

class LoginPage:
    def __init__(self, root, on_login):
        self.root = root
        self.on_login = on_login
        
        # Center frame with reduced size
        self.frame = ctk.CTkFrame(
            root,
            fg_color="#ffffff",
            corner_radius=15,
            width=400,  # Reduced from 700
            height=500   # Reduced from 600
        )
        self.frame.place(relx=0.5, rely=0.5, anchor="center")
        
        # Title
        ctk.CTkLabel(
            self.frame,
            fg_color="#ffffff",
            text="Login",
            font=("SF Pro Display", 32, "bold"),
            text_color="#000000"
        ).pack(pady=(30, 40))
        
        # Form container
        form_frame = ctk.CTkFrame(
            self.frame,
            fg_color="#ffffff",
            width=400
        )
        form_frame.pack(pady=0)
        
        # Username with icon
        self.username = ctk.CTkEntry(
            form_frame,
            placeholder_text="Username",
            text_color="#000000",
            fg_color="#ffffff",
            font=("SF Pro Display", 14),
            width=400,
            height=40,
            border_width=1,
            corner_radius=8
        )
        self.username.pack(pady=(0, 30))
        
        # Password with icon
        self.password = ctk.CTkEntry(
            form_frame,
            placeholder_text="Password",
            show="•",
            text_color="#000000",
            fg_color="#ffffff",
            font=("SF Pro Display", 14),
            width=400,
            height=40,
            border_width=1,
            corner_radius=8
        )
        self.password.pack(pady=(0, 30))

        # Login button
        self.login_btn = ctk.CTkButton(
            form_frame,
            text="Login",
            font=("SF Pro Display", 14, "bold"),
            width=400,
            height=40,
            corner_radius=8,
            fg_color="#007AFF",
            hover_color="#0062CC"
        )
        self.login_btn.configure(command=self.login)
        self.login_btn.pack(pady=(0, 30))
        
        # Error label
        self.error_label = ctk.CTkLabel(
            form_frame,
            text="",
            text_color="#FF3B30",
            font=("SF Pro Display", 12)
        )
        self.error_label.pack(pady=(0, 20))
        
    def login(self):
        username = self.username.get()
        password = self.password.get()
        
        if not username or not password:
            self.error_label.configure(text="Please enter username and password")
            return
            
        # Create basic auth header
        auth = base64.b64encode(
            f"{username}:{password}".encode()
        ).decode()
        
        try:
            response = requests.post(
                f"{BACKEND_URL}/login",
                headers={"Authorization": f"Basic {auth}"}
            )
            
            if response.status_code == 200:
                user_data = response.json()
                self.frame.destroy()
                self.on_login(user_data, auth)
            else:
                self.error_label.configure(text="Invalid credentials")
        except Exception as e:
            self.error_label.configure(text="Login failed")

class ImageViewerDialog(ctk.CTkToplevel):
    def __init__(self, parent, augmented_path, derained_path):
        super().__init__(parent)
        self.title("Recognition Images")
        self.geometry("800x400")
        
        # Set dialog background color
        self.configure(fg_color="#ffffff")

        # Create image frames with white background
        aug_frame = ctk.CTkFrame(self, fg_color="#ffffff")
        aug_frame.pack(side="left", expand=True, fill="both", padx=10, pady=10)
        derain_frame = ctk.CTkFrame(self, fg_color="#ffffff")
        derain_frame.pack(side="right", expand=True, fill="both", padx=10, pady=10)

        # Load and display images
        aug_img = Image.open(augmented_path)
        derain_img = Image.open(derained_path)
        
        aug_img = aug_img.resize((256, 256), Image.Resampling.LANCZOS)
        derain_img = derain_img.resize((256, 256), Image.Resampling.LANCZOS)
        
        aug_imgtk = ctk.CTkImage(light_image=aug_img, dark_image=aug_img, size=(256, 256))
        derain_imgtk = ctk.CTkImage(light_image=derain_img, dark_image=derain_img, size=(256, 256))

        ctk.CTkLabel(aug_frame, text="Rain Augmented").pack()
        ctk.CTkLabel(aug_frame, image=aug_imgtk).pack()
        
        ctk.CTkLabel(derain_frame, text="De-Rained").pack()
        ctk.CTkLabel(derain_frame, image=derain_imgtk).pack()

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        
        # Configure for fullscreen windowed mode
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}+0+0")
        self.root.state('zoomed')
        self.root.resizable(True, True)
        self.root.minsize(1600, 900)
        
        # Set root background color to white
        self.root.configure(fg_color="#ffffff")
        
        # Initialize session first
        self.session = requests.Session()
        
        # Start with login
        self.login_page = LoginPage(root, self.on_login)
        
        # Initialize frame queues
        self.frame_queue = queue.Queue(maxsize=20)  # For processing pipeline
        self.display_queue = queue.Queue(maxsize=20)  # For live display
        self.queue_active = False
        self.display_active = False

        self.loading = None  # Add this line
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load the generator model
        try:
            self.generator = torch.jit.load("models/generator_model.pt", map_location=self.device)
            self.generator.eval()
            print("✅ Generator model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading generator model: {e}")
            self.generator = None

    def show_no_signal_message(self, label_name):
        """Display no signal message on video label"""
        try:
            if not hasattr(self, label_name) or getattr(self, label_name) is None:
                return
                
            # Create a black image with text
            img = Image.new('RGB', (600, 450), color='black')
            d = ImageDraw.Draw(img)
            try:
                # Try default system font if arial.ttf not found
                d.text((300, 225), "No Signal\nWaiting for camera...", 
                       fill='white', anchor="mm", align="center",
                       font=ImageFont.truetype("arial.ttf", 36))
            except OSError:
                # Fallback to default font
                d.text((300, 225), "No Signal\nWaiting for camera...", 
                       fill='white', anchor="mm", align="center")
            
            imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(600, 450))
            label = getattr(self, label_name)
            if label and label.winfo_exists():
                try:
                    label.configure(image=imgtk)
                    label.image = imgtk  # Keep reference
                except Exception:
                    pass
        except Exception as e:
            print(f"Error showing no signal message: {e}")

    def on_login(self, user_data, auth):
        """Called after successful login"""
        self.user = user_data
        self.session.headers["Authorization"] = f"Basic {auth}"
        
        # Show loading screen
        self.loading = LoadingScreen(self.root)
        
        # Initialize in background
        self.root.after(100, self.initialize_system)

    def initialize_system(self):
        """Initialize system components with loading screen"""
        try:
            # Initialize rest of the app
            init_db()
            init_assets()
            self.register_running = True
            self.recognition_logs = []
            self.log_container = None
            
            # Configure theme and styles
            ctk.set_appearance_mode("light")
            self.title_font = ("SF Pro Display", 20, "bold")
            self.text_font = ("SF Pro Text", 12)
            self.small_font = ("SF Pro Text", 10)
            
            self.bg_color = "#ffffff"
            self.accent_color = "#8ac4ff"
            self.text_color = "#000000"
            self.secondary_color = "#f5f5f7"
            
            self.setup_ui()
            self.init_camera()
            
            # Load models and resources
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.last_processed_time = 0
            self.processing_interval = 0.1

            # Load rain patches
            self.rain_patches = []
            rain_patches_dir = "extracted_rain_patches/heavy"
            for filename in os.listdir(rain_patches_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    patch_path = os.path.join(rain_patches_dir, filename)
                    patch = cv2.imread(patch_path)
                    if patch is not None:
                        self.rain_patches.append(patch)

            # Initialize threads
            self.threads = {
                'live': None,
                'recognition': None,
                'registration': None,
                'analytics': None,
                'live_recognition': None
            }
            self.thread_running = {
                'live': False,
                'recognition': False,
                'registration': False,
                'analytics': False,
                'live_recognition': False
            }
            
            # Start all threads
            self.start_all_threads()
            
            # Remove loading screen properly
            if hasattr(self, 'loading') and self.loading:
                self.loading.destroy()
                self.loading = None
                
            # Show initial view
            self.show_file_derain()
        except Exception as e:
            print(f"Initialization error: {e}")

    def start_all_threads(self):
        """Initialize and start all UI threads"""
        # Live feed thread
        self.thread_running['live'] = True
        self.threads['live'] = threading.Thread(
            target=self.update_live_feed,
            daemon=True
        )
        self.threads['live'].start()
        
        # Recognition thread
        self.thread_running['recognition'] = True
        self.threads['recognition'] = threading.Thread(
            target=self.update_processed_output,
            daemon=True
        )
        self.threads['recognition'].start()
        
        # Registration thread
        self.thread_running['registration'] = True
        self.threads['registration'] = threading.Thread(
            target=self.update_register_feed,
            daemon=True
        )
        self.threads['registration'].start()

        # Add live recognition thread
        self.thread_running['live_recognition'] = True
        self.threads['live_recognition'] = threading.Thread(
            target=self.process_live_recognition,
            daemon=True
        )
        self.threads['live_recognition'].start()

    def process_frame_for_model(self, frame):
        """Process frame for model input to match training/testing pipeline"""
        frame = cv2.resize(frame, (256, 256))
        # Convert to float and normalize to [0,1]
        frame = frame.astype(np.float32) / 255.0
        # Convert to tensor and add batch dimension
        frame = torch.FloatTensor(frame).permute(2, 0, 1).unsqueeze(0)
        return frame.to(self.device)

    def apply_model(self, input_tensor):
        """Apply derain model with consistent preprocessing"""
        try:
            if self.generator is None:
                raise ValueError("Generator model not loaded")
                
            with torch.no_grad():
                # Apply model
                output = self.generator(input_tensor)
                # Ensure output size is correct
                output = torch.nn.functional.interpolate(output, size=(256, 256))
                # Convert to image format
                output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
                output = (np.clip(output, 0, 1) * 255).astype(np.uint8)
                return output
        except Exception as e:
            print(f"Error in model inference: {e}")
            return None

    def detect_faces(self, frame):
        """Detect faces in frame and return frame with rectangles"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        # Draw rectangles on a copy of the frame
        frame_with_faces = frame.copy()
        faces_detected = len(faces) > 0
        
        for (x, y, w, h) in faces:
            cv2.rectangle(
                frame_with_faces, 
                (x, y), 
                (x+w, y+h), 
                (0, 255, 0), 
                2
            )
            
        return frame_with_faces, faces_detected

    def process_frame_for_display(self, frame, size=(800, 600)):
        """Process frame for display at larger size"""
        image = Image.fromarray(frame)
        return ctk.CTkImage(light_image=image, dark_image=image, size=size)

    def update_processed_output(self):
        while self.thread_running['recognition']:
            try:
                if not self.queue_active or self.frame_queue.empty():
                    time.sleep(0.1)
                    continue

                # Get and process frame
                frame = self.frame_queue.get()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_with_faces, faces_detected = self.detect_faces(frame_rgb)
                
                # Show original feed
                display_img = self.process_frame_for_display(frame_with_faces)
                if hasattr(self, 'video_label1'):
                    self.video_label1.configure(image=display_img)
                    self.video_label1.image = display_img

                if not faces_detected:
                    if hasattr(self, 'recognition_label'):
                        self.recognition_label.configure(text="Awaiting face detection...", text_color=self.text_color)
                    continue

                try:
                    if hasattr(self, 'is_direct_mode') and self.is_direct_mode:
                        # Direct derain mode - consistent with test pipeline
                        model_input = self.process_frame_for_model(frame_rgb)
                        processed_frame = self.apply_model(model_input)
                    else:
                        # Rain augmented mode
                        model_frame = cv2.resize(frame_rgb, (256, 256))
                        augmented_frame = generate_rainy_image(model_frame, self.rain_patches, "heavy")
                        model_input = self.process_frame_for_model(augmented_frame)
                        processed_frame = self.apply_model(model_input)
                        
                        # Display augmented frame
                        augmented_img = self.process_frame_for_display(augmented_frame)
                        self.video_label2.configure(image=augmented_img)
                        self.video_label2.image = augmented_img

                    # Show derained result
                    if processed_frame is not None:
                        display_img = self.process_frame_for_display(processed_frame)
                        if hasattr(self, 'video_label3'):
                            self.video_label3.configure(image=display_img)
                            self.video_label3.image = display_img

                        # Face recognition
                        try:
                            img_byte_arr = BytesIO()
                            Image.fromarray(processed_frame).save(img_byte_arr, format='PNG')
                            files = {'file': ('image.png', img_byte_arr.getvalue(), 'image/png')}
                            
                            recognition_response = self.session.post(
                                f"{BACKEND_URL}/recognize_face",
                                files=files
                            )
                            
                            if recognition_response.status_code == 200:
                                result = recognition_response.json()
                                name = result["name"]
                                confidence = result["confidence"] * 100
                                if hasattr(self, 'recognition_label'):
                                    self.recognition_label.configure(
                                        text=f"Recognized: {name}\nConfidence: {confidence:.2f}%",
                                        text_color="green" if confidence > 50 else "orange"
                                    )
                                
                                # Log recognition results - for both known and unknown faces
                                orig_img = Image.fromarray(frame_rgb)
                                derain_img = Image.fromarray(processed_frame)
                                self.add_recognition_log(name, confidence, orig_img, derain_img)
                            else:
                                if hasattr(self, 'recognition_label'):
                                    self.recognition_label.configure(text="Recognition Failed", text_color="red")
                                # Log failed recognition attempt
                                orig_img = Image.fromarray(frame_rgb)
                                derain_img = Image.fromarray(processed_frame)
                                self.add_recognition_log("Unknown", 0.0, orig_img, derain_img)
                        except Exception as e:
                            print(f"Recognition error: {e}")
                            if hasattr(self, 'recognition_label'):
                                self.recognition_label.configure(text="Recognition Failed", text_color="red")
                            # Log error case
                            try:
                                orig_img = Image.fromarray(frame_rgb)
                                derain_img = Image.fromarray(processed_frame)
                                self.add_recognition_log("Error", 0.0, orig_img, derain_img)
                            except:
                                pass

                except Exception as e:
                    print(f"Processing error: {e}")

            except Exception as e:
                print(f"Error in processed output: {e}")
                time.sleep(0.1)

    def process_live_recognition(self):
        """Separate thread for live view recognition"""
        while self.thread_running['live']:
            try:
                if not self.display_active or self.display_queue.empty():
                    time.sleep(0.1)
                    continue

                frame_data = self.display_queue.get()
                frame = frame_data['frame']
                
                # Only process if frame is recent (within last 0.5 seconds)
                if time.time() - frame_data['timestamp'] > 0.5:
                    continue

                # Process for recognition
                frame_for_recog = Image.fromarray(frame)
                img_byte_arr = BytesIO()
                frame_for_recog.save(img_byte_arr, format='PNG')
                files = {'file': ('image.png', img_byte_arr.getvalue(), 'image/png')}
                
                recognition_response = self.session.post(
                    f"{BACKEND_URL}/recognize_face",
                    files=files
                )
                
                if recognition_response.status_code == 200:
                    result = recognition_response.json()
                    name = result["name"]
                    confidence = result["confidence"] * 100
                    if hasattr(self, 'live_recognition_label'):
                        self.live_recognition_label.configure(
                            text=f"Recognized: {name}\nConfidence: {confidence:.2f}%",
                            text_color="green" if confidence > 50 else "orange"
                        )
                else:
                    if hasattr(self, 'live_recognition_label'):
                        self.live_recognition_label.configure(
                            text="Unknown Person", 
                            text_color="red"
                        )
                        
            except Exception as e:
                print(f"Error in live recognition: {e}")
                time.sleep(0.1)

    def add_recognition_log(self, name, confidence, original_img=None, derained_img=None):
        """Modified to save original and derained images"""
        if not self.log_container:
            return
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save to database and get log ID
        log_id = save_recognition(timestamp, name, confidence, original_img, derained_img)
        
        # Create new log entry immediately
        self.create_log_entry(log_id, timestamp, name, confidence)
        self.refresh_logs()  # Refresh the logs display

    def create_log_entry(self, log_id, timestamp, name, confidence):
        """Create a log entry widget"""
        if not self.log_container:
            return
            
        log_entry = ctk.CTkFrame(
            self.log_container,
            fg_color=self.secondary_color,
            corner_radius=5
        )
        log_entry.log_id = log_id
        log_entry.pack(fill="x", pady=5, padx=5)
        
        log_text = f"Time: {timestamp}\nPerson: {name}\nConfidence: {confidence:.2f}%"
        log_label = ctk.CTkLabel(
            log_entry,
            text=log_text,
            font=self.text_font,
            text_color=self.text_color,
            cursor="hand2"
        )
        log_label.pack(pady=5, padx=5)
        
        # Bind click event
        log_label.bind("<Button-1>", lambda e, id=log_id: self.show_log_images(id))
        
        if len(self.recognition_logs) > 50:
            oldest_log = self.recognition_logs.pop(0)
            oldest_log.destroy()
        self.recognition_logs.append(log_entry)

    def show_log_images(self, log_id):
        log_data = get_recognition(log_id)
        if log_data:
            dialog = ImageViewerDialog(
                self.root,
                log_data['augmented_path'],
                log_data['derained_path']
            )
            dialog.focus()

    def init_camera(self):
        """Initialize camera label attributes"""
        # Remove cv2.VideoCapture usage
        self.video_label1 = None
        self.video_label2 = None 
        self.video_label3 = None
        self.register_video_label = None
        self.current_frame = None

    def setup_ui(self):
        # Main container
        self.main_container = ctk.CTkFrame(
            self.root,
            fg_color=self.bg_color,
            corner_radius=0
        )
        self.main_container.pack(fill="both", expand=True, padx=5, pady=0)

        # Sidebar setup
        self.sidebar = ctk.CTkFrame(
            self.main_container,
            fg_color=self.bg_color,
            width=80,
            corner_radius=0
        )
        self.sidebar.pack(side="left", fill="y", padx=(0, 0))

        # Navigation buttons
        self.nav_buttons = [
            ("", "upload.png", self.show_file_derain),  
            ("", "face1.png", self.show_live_feed),
            ("", "face2.png", self.show_direct_derain),
            # ("", "face3.png", self.show_direct_recognition),  # Add this new button
            ("", "add.png", self.show_register_face),
            ("", "chart.png", self.show_analytics),
            ("", "settings.png", self.show_settings)
        ]

        for i, (text, icon, command) in enumerate(self.nav_buttons):
            btn = self.create_nav_button(text, icon, command)
            btn.pack(pady=(20 if i == 0 else 10))

        # Content area
        self.content_area = ctk.CTkFrame(
            self.main_container,
            fg_color=self.bg_color,
            corner_radius=0
        )
        self.content_area.pack(side="right", fill="both", expand=True)

    def create_nav_button(self, text, icon_name, command):
        btn_frame = ctk.CTkFrame(
            self.sidebar,
            fg_color="transparent"
        )
        
        # Load and resize icon with detailed error handling
        try:
            icon_path = ICONS_PATH / icon_name
            if not icon_path.exists():
                print(f"Warning: Icon not found at {icon_path}")
                # Create a colored square as fallback
                fallback = Image.new('RGB', (24, 24), self.accent_color)
                # Draw text on fallback icon
                draw = ImageDraw.Draw(fallback)
                draw.text((12, 12), text[0], fill='white', anchor='mm')
                icon = fallback
            else:
                icon = Image.open(icon_path).convert('RGBA')
                # Ensure icon is square
                icon = icon.resize((24, 24), Image.Resampling.LANCZOS)
            
            icon_ctk = ctk.CTkImage(light_image=icon, dark_image=icon, size=(24, 24))
            
        except Exception as e:
            print(f"Error loading icon {icon_name}: {str(e)}")
            fallback = Image.new('RGB', (24, 24), self.accent_color)
            # Draw first letter of text on fallback
            draw = ImageDraw.Draw(fallback)
            draw.text((12, 12), text[0], fill='white', anchor='mm')
            icon_ctk = ctk.CTkImage(light_image=fallback, dark_image=fallback, size=(24, 24))
        
        btn = ctk.CTkButton(
            btn_frame,
            text="",
            image=icon_ctk,
            command=command,
            width=60,
            height=60,
            fg_color="transparent",
            hover_color=self.accent_color,
            corner_radius=0
        )
        btn.pack(pady=5)
        
        label = ctk.CTkLabel(
            btn_frame,
            text=text,
            font=self.small_font,
            text_color=self.text_color
        )
        label.pack()
        
        return btn_frame

    
    # def show_recognition(self):
    #     """Show recognition view with rain augmentation"""
    #     self.clear_screen()
    #     self.live_active = False
    #     self.is_direct_mode = False  # Add this line
    #     self.queue_active = True  # Enable queue when showing recognition
    #     self.display_active = True  # Enable display feed
        
    #     # Header with increased bottom padding
    #     header = ctk.CTkLabel(
    #         self.content_area,
    #         text="Enhanced Face Recognition",
    #         font=self.title_font,
    #         text_color=self.text_color
    #     )
    #     header.pack(pady=(30, 20))  # Increased bottom padding
        
    #     # Video container with grid layout
    #     self.video_container = ctk.CTkFrame(
    #         self.content_area,
    #         fg_color=self.bg_color,
    #         corner_radius=0
    #     )
    #     self.video_container.pack(expand=True)
        
    #     frames = [
    #         ("Original Feed", "video_label1"),
    #         ("Rain Augmented", "video_label2"),
    #         ("De-Rained Result", "video_label3")
    #     ]
        
    #     for i, (title, label_name) in enumerate(frames):
    #         frame = self.create_video_frame(title)
    #         frame.grid(row=0, column=i, padx=10)
    #         label = ctk.CTkLabel(frame, text="")
    #         label.pack(expand=True, fill="both")
    #         setattr(self, label_name, label)
        
    #     # Recognition result label with optimal spacing
    
    def create_video_frame(self, text):
        frame = ctk.CTkFrame(
            self.video_container,
            fg_color="#ffffff",  # Changed from secondary_color to white
            corner_radius=0,
            width=600,  # Increased from 400
            height=600  # Increased from 400
        )
        frame.pack_propagate(False)
        
        title = ctk.CTkLabel(
            frame,
            text=text,
            font=self.text_font,
            text_color=self.text_color
        )
        title.pack(pady=(10, 5))
        
        return frame

    def show_live_feed(self):
        self.clear_screen()
        self.queue_active = False
        self.display_active = True  # Enable live processing
        
        # Container frame
        container = ctk.CTkFrame(self.content_area, fg_color=self.bg_color)
        container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header
        header = ctk.CTkLabel(
            container,
            text="Live Face Recognition",
            font=self.title_font,
            text_color=self.text_color
        )
        header.pack(pady=(0, 20))
        
        # Video frame
        frame = ctk.CTkFrame(
            container,
            fg_color=self.secondary_color,
            corner_radius=0
        )
        frame.pack(expand=True)
        
        self.video_label1 = ctk.CTkLabel(frame, text="")
        self.video_label1.pack(expand=True, padx=40, pady=40)
        
        # Recognition result label
        self.live_recognition_label = ctk.CTkLabel(
            container,
            text="Awaiting face detection...",
            font=self.title_font,
            text_color=self.text_color
        )
        self.live_recognition_label.pack(pady=20)

    def show_direct_recognition(self):
        """Show direct face recognition without deraining"""
        self.clear_screen()
        self.queue_active = False
        self.display_active = False
        
        # Header
        header = ctk.CTkLabel(
            self.content_area,
            text="Direct Face Recognition",
            font=self.title_font,
            text_color=self.text_color
        )
        header.pack(pady=(30, 20))
        
        # Upload button and status
        control_frame = ctk.CTkFrame(
            self.content_area,
            fg_color=self.bg_color,
            corner_radius=0
        )
        control_frame.pack(pady=10)
        
        def process_face():
            file_path = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[("Image files", "*.png *.jpg *.jpeg")]
            )
            if not file_path:
                return
                
            try:
                # Load and display original image
                img = Image.open(file_path).convert('RGB')
                img_array = np.array(img)
                
                # Show original image
                orig_display = self.process_frame_for_display(img_array)
                self.file_label1.configure(image=orig_display)
                self.file_label1.image = orig_display
                
                # Send directly for face recognition
                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format='PNG')
                files = {'file': ('image.png', img_byte_arr.getvalue(), 'image/png')}
                
                recognition_response = self.session.post(
                    f"{BACKEND_URL}/recognize_face",
                    files=files
                )
                
                if recognition_response.status_code == 200:
                    result = recognition_response.json()
                    name = result["name"]
                    confidence = result["confidence"] * 100
                    status_label.configure(
                        text=f"Recognized: {name}\nConfidence: {confidence:.2f}%",
                        text_color="green" if confidence > 50 else "orange"
                    )
                else:
                    status_label.configure(text="Recognition failed", text_color="red")
                    
            except Exception as e:
                status_label.configure(text=f"Error: {str(e)}", text_color="red")
        
        upload_btn = ctk.CTkButton(
            control_frame,
            text="Select Image",
            command=process_face,
            width=200
        )
        upload_btn.pack(side="left", padx=10)
        
        status_label = ctk.CTkLabel(
            control_frame,
            text="Select an image to process",
            font=self.text_font,
            text_color=self.text_color
        )
        status_label.pack(side="left", padx=10)
        
        # Image display container
        self.video_container = ctk.CTkFrame(
            self.content_area,
            fg_color=self.bg_color,
            corner_radius=0
        )
        self.video_container.pack(expand=True)
        
        # Create image frame
        frame = self.create_video_frame("Input Image")
        frame.pack(expand=True)
        
        # Image label
        self.file_label1 = ctk.CTkLabel(frame, text="No image loaded")
        self.file_label1.pack(expand=True, fill="both")

    def show_direct_derain(self):
        """Show direct derain recognition view without rain augmentation"""
        self.clear_screen()
        self.live_active = False
        self.is_direct_mode = True  # Add this flag
        self.queue_active = True
        self.display_active = True
        
        # Header with increased bottom padding
        header = ctk.CTkLabel(
            self.content_area,
            text="Rain-Free Face Recognition",
            font=self.title_font,
            text_color=self.text_color
        )
        header.pack(pady=(30, 20))
        
        # Video container with grid layout
        self.video_container = ctk.CTkFrame(
            self.content_area,
            fg_color=self.bg_color,
            corner_radius=0
        )
        self.video_container.pack(expand=True)
        
        frames = [
            ("Original Feed", "video_label1"),
            ("De-Rained Result", "video_label3")
        ]
        
        for i, (title, label_name) in enumerate(frames):
            frame = self.create_video_frame(title)
            frame.grid(row=0, column=i, padx=20)
            label = ctk.CTkLabel(frame, text="")
            label.pack(expand=True, fill="both")
            setattr(self, label_name, label)
        
        # Recognition result label with optimal spacing
        self.recognition_label = ctk.CTkLabel(
            self.content_area,
            text="Awaiting face detection...",
            font=self.title_font,
            text_color=self.text_color
        )
        self.recognition_label.pack(pady=(5, 180))

    def show_file_derain(self):
        """Show file upload and processing screen"""
        self.clear_screen()
        self.queue_active = False
        self.display_active = False
        
        # Header
        header = ctk.CTkLabel(
            self.content_area,
            text="Process Image File",
            font=self.title_font,
            text_color=self.text_color
        )
        header.pack(pady=(30, 20))
        
        # Upload button and status
        control_frame = ctk.CTkFrame(
            self.content_area,
            fg_color=self.bg_color,
            corner_radius=0
        )
        control_frame.pack(pady=10)
        
        def process_file():
            file_path = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[("Image files", "*.png *.jpg *.jpeg")]
            )
            if not file_path:
                return
                
            try:
                # Load and display original image
                img = Image.open(file_path).convert('RGB')
                img_array = np.array(img)
                
                # Show original image
                orig_display = self.process_frame_for_display(img_array)
                self.file_label1.configure(image=orig_display)
                self.file_label1.image = orig_display
                
                # Process with derain model using same preprocessing as direct_derain
                model_input = self.process_frame_for_model(img_array)
                processed_frame = self.apply_model(model_input)
                
                if processed_frame is not None:
                    status_label.configure(text="Processing complete", text_color="green")
                    
                    # Display processed result
                    result_display = self.process_frame_for_display(processed_frame)
                    self.file_label2.configure(image=result_display)
                    self.file_label2.image = result_display
                    
                    # Try recognition on processed frame
                    img_byte_arr = BytesIO()
                    Image.fromarray(processed_frame).save(img_byte_arr, format='PNG')
                    files = {'file': ('image.png', img_byte_arr.getvalue(), 'image/png')}
                    
                    recog_response = self.session.post(
                        f"{BACKEND_URL}/recognize_face",
                        files=files
                    )
                    
                    if recog_response.status_code == 200:
                        result = recog_response.json()
                        name = result["name"]
                        confidence = result["confidence"] * 100
                        status_label.configure(
                            text=f"Recognized: {name}\nConfidence: {confidence:.2f}%",
                            text_color="green" if confidence > 50 else "orange"
                        )
                        
                        # Log the result
                        self.add_recognition_log(
                            name,
                            confidence,
                            Image.fromarray(img_array),
                            Image.fromarray(processed_frame)
                        )
                    else:
                        status_label.configure(text="Recognition failed", text_color="red")
                else:
                    status_label.configure(text="Processing failed", text_color="red")
                    
            except Exception as e:
                status_label.configure(text=f"Error: {str(e)}", text_color="red")
        
        upload_btn = ctk.CTkButton(
            control_frame,
            text="Select Image",
            command=process_file,
            width=200
        )
        upload_btn.pack(side="left", padx=10)
        
        status_label = ctk.CTkLabel(
            control_frame,
            text="Select an image to process",
            font=self.text_font,
            text_color=self.text_color
        )
        status_label.pack(side="left", padx=10)
        
        # Image display container
        self.video_container = ctk.CTkFrame(
            self.content_area,
            fg_color=self.bg_color,
            corner_radius=0
        )
        self.video_container.pack(expand=True)
        
        # Create image frames
        frames = [
            ("Original Image", "file_label1"),
            ("De-Rained Result", "file_label2")
        ]
        
        for i, (title, label_name) in enumerate(frames):
            frame = self.create_video_frame(title)
            frame.grid(row=0, column=i, padx=20)
            label = ctk.CTkLabel(frame, text="")
            label.pack(expand=True, fill="both")
            setattr(self, label_name, label)

    def show_analytics(self):
        self.clear_screen()
        self.queue_active = False  # Disable queue for other views
        self.display_active = False  # Disable display feed
        
        header = ctk.CTkLabel(
            self.content_area,
            text="Recognition Logs & Analytics",
            font=self.title_font
        )
        header.pack(pady=20)

        # Filter controls container
        filter_frame = ctk.CTkFrame(self.content_area)
        filter_frame.pack(pady=10, padx=20, fill="x")

        # Date range filters
        date_frame = ctk.CTkFrame(filter_frame, fg_color="transparent")
        date_frame.pack(side="left", fill="x", expand=True)
        
        from_date = ctk.CTkEntry(date_frame, placeholder_text="From (YYYY-MM-DD)")
        from_date.pack(side="left", padx=5)
        
        to_date = ctk.CTkEntry(date_frame, placeholder_text="To (YYYY-MM-DD)")
        to_date.pack(side="left", padx=5)

        filter_btn = ctk.CTkButton(
            date_frame, 
            text="Filter", 
            command=lambda: self.filter_logs(from_date.get(), to_date.get())
        )
        filter_btn.pack(side="left", padx=5)

        reset_btn = ctk.CTkButton(
            date_frame, 
            text="Reset", 
            command=self.load_existing_logs
        )
        reset_btn.pack(side="left", padx=5)

        # Show Unknown checkbox
        self.show_unknown_var = ctk.BooleanVar(value=True)
        show_unknown_cb = ctk.CTkCheckBox(
            filter_frame,
            text="Show Unknown Faces",
            variable=self.show_unknown_var,
            command=self.refresh_logs,
            font=self.text_font
        )
        show_unknown_cb.pack(side="right", padx=20)

        # Create log container
        self.create_log_container()
        
        # Load existing logs
        self.load_existing_logs()

    def refresh_logs(self):
        """Refresh logs based on current filter settings"""
        self.load_existing_logs()

    def load_existing_logs(self):
        """Load and display existing logs from database"""
        # Clear existing logs
        for widget in self.log_container.winfo_children():
            widget.destroy()
        self.recognition_logs = []

        logs = get_all_logs()
        for log in logs:
            # Skip unknown faces if checkbox is unchecked
            if not self.show_unknown_var.get() and log['person_name'].lower() == 'unknown':
                continue
                
            self.create_log_entry(
                log['id'],
                log['timestamp'],
                log['person_name'],
                log['confidence']
            )

    def filter_logs(self, from_date, to_date):
        """Filter logs by date range"""
        try:
            # Clear existing logs
            for widget in self.log_container.winfo_children():
                widget.destroy()
            self.recognition_logs = []

            logs = get_all_logs()
            
            for log in logs:
                # Skip unknown faces if checkbox is unchecked
                if not self.show_unknown_var.get() and log['person_name'].lower() == 'unknown':
                    continue
                    
                log_date = datetime.strptime(log['timestamp'], "%Y-%m-%d %H:%M:%S").date()
                try:
                    from_date = datetime.strptime(from_date, "%Y-%m-%d").date()
                    to_date = datetime.strptime(to_date, "%Y-%m-%d").date()
                    
                    if from_date <= log_date <= to_date:
                        self.create_log_entry(
                            log['id'],
                            log['timestamp'],
                            log['person_name'],
                            log['confidence']
                        )
                except ValueError:
                    continue
        except Exception as e:
            print(f"Error filtering logs: {e}")

    def create_log_container(self):
        log_frame = ctk.CTkFrame(
            self.content_area,
            fg_color="#ffffff",
            corner_radius=0
        )
        log_frame.pack(fill="both", expand=True, pady=20, padx=20)
        
        # Log header
        log_header = ctk.CTkLabel(
            log_frame,
            text="Recognition Logs",
            font=self.title_font,
            text_color=self.text_color
        )
        log_header.pack(pady=10)
        
        # Scrollable log container
        self.log_container = ctk.CTkScrollableFrame(
            log_frame,
            fg_color="#ffffff",
            corner_radius=0
        )
        self.log_container.pack(fill="both", expand=True, padx=10, pady=10)

    def clear_screen(self):
        """Modified to preserve threads"""
        for widget in self.content_area.winfo_children():
            widget.destroy()

    def show_register_face(self):
        self.clear_screen()
        self.queue_active = False
        self.display_active = True  # Changed to True to enable video

        # Main header only
        header = ctk.CTkLabel(
            self.content_area,
            text="Person Registration",
            font=self.title_font,
            text_color=self.text_color
        )
        header.pack(pady=(20, 30))

        # Create two columns with white background
        left_frame = ctk.CTkFrame(self.content_area, fg_color="#ffffff")
        left_frame.pack(side="left", fill="both", expand=True, padx=20, pady=20)
        
        right_frame = ctk.CTkFrame(self.content_area, fg_color="#ffffff")
        right_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

        # Camera feed moved to top with proper sizing
        self.live_register_frame = ctk.CTkFrame(
            left_frame, 
            fg_color="#ffffff",
            width=640,   # Fixed width
            height=480  # Fixed height
        )
        self.live_register_frame.pack_propagate(False)  # Prevent frame from shrinking
        self.live_register_frame.pack(pady=(0, 20))
        
        # Initialize video label with proper size
        self.register_video_label = ctk.CTkLabel(
            self.live_register_frame,
            text="",
            width=640,
            height=480
        )
        self.register_video_label.pack(expand=True)

        # Left column - Registration form centered below camera
        form_frame = ctk.CTkFrame(left_frame, fg_color="#ffffff")
        form_frame.pack(pady=20)

        # Form fields with consistent spacing and centering
        fields = [
            ("ID:", self.text_font, "id_entry"),
            ("Full Name:", self.text_font, "name_entry")
        ]

        form_width = 400  # Increased form width
        for label_text, font, entry_name in fields:
            # Container for each field pair
            field_frame = ctk.CTkFrame(form_frame, fg_color="#ffffff", width=form_width)
            field_frame.pack(fill="x", pady=5)
            
            # Label
            label = ctk.CTkLabel(
                field_frame, 
                text=label_text, 
                font=font,
                width=100,  # Fixed width for alignment
                anchor="e"
            )
            label.pack(side="left", padx=(0, 10))
            
            # Entry
            entry = ctk.CTkEntry(field_frame, width=250)  # Increased width
            entry.pack(side="left")
            setattr(self, entry_name, entry)

        # Add upload section
        upload_frame = ctk.CTkFrame(form_frame, fg_color="#ffffff")
        upload_frame.pack(pady=10)

        def handle_image_upload():
            file_path = filedialog.askopenfilename(
                title="Select Image",
                filetypes=[("Image files", "*.png *.jpg *.jpeg")]
            )
            if not file_path:
                return

            try:
                # Load and display image
                img = Image.open(file_path).convert('RGB')
                img_array = np.array(img)
                display_img = self.process_frame_for_display(img_array, size=(640, 480))
                self.register_video_label.configure(image=display_img)
                self.register_video_label.image = display_img
                
                # Store image for registration
                self.current_frame = img_array
                self.feedback_label.configure(text="Image loaded successfully", text_color="green")
            except Exception as e:
                self.feedback_label.configure(text=f"Error loading image: {e}", text_color="red")

        upload_btn = ctk.CTkButton(
            upload_frame,
            text="Upload Image",
            command=handle_image_upload,
            width=350
        )
        upload_btn.pack(pady=5)

        separator = ctk.CTkLabel(
            upload_frame,
            text="- OR -",
            font=self.text_font
        )
        separator.pack(pady=5)

        camera_label = ctk.CTkLabel(
            upload_frame,
            text="Use Camera Feed",
            font=self.text_font
        )
        camera_label.pack(pady=5)

        # Register button aligned with fields
        capture_button = ctk.CTkButton(
            form_frame,
            text="Register Face",
            command=self.capture_face_embedding,
            width=350  # Match form width
        )
        capture_button.pack(pady=20)

        # Feedback label
        self.feedback_label = ctk.CTkLabel(
            form_frame, 
            text="",
            font=self.text_font
        )
        self.feedback_label.pack(pady=10)

        # Right column - Registered Users with cleaner layout
        # Search frame with better spacing
        search_frame = ctk.CTkFrame(right_frame, fg_color="#ffffff")
        search_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        self.search_entry = ctk.CTkEntry(
            search_frame, 
            placeholder_text="Search by ID or name...",
            width=200
        )
        self.search_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        search_btn = ctk.CTkButton(
            search_frame,
            text="Search",
            command=lambda: self.load_registered_users(self.search_entry.get())
        )
        search_btn.pack(side="right")

        # Users container with more breathing room
        self.users_container = ctk.CTkScrollableFrame(
            right_frame,
            fg_color="#ffffff",
            corner_radius=0
        )
        self.users_container.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        # Load registered users
        self.load_registered_users()

    def update_register_feed(self):
        """Update registration camera feed"""
        no_video_shown = False
        last_frame_time = time.time()
        
        while self.thread_running['registration']:
            try:
                # Skip if registration view is not active
                if not hasattr(self, 'register_video_label') or self.register_video_label is None:
                    time.sleep(0.3)
                    continue
                    
                response = self.session.get(f"{BACKEND_URL}/get_latest_frame")
                if response.status_code == 200:
                    frame_bytes = response.content
                    frame = cv2.imdecode(
                        np.frombuffer(frame_bytes, np.uint8),
                        cv2.IMREAD_COLOR
                    )
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Store frame for face registration
                    self.current_frame = frame_rgb
                    
                    # Update display only if label still exists
                    try:
                        if self.register_video_label.winfo_exists():
                            image = Image.fromarray(frame_rgb)
                            imgtk = ctk.CTkImage(
                                light_image=image,
                                dark_image=image,
                                size=(640, 480)
                            )
                            self.register_video_label.configure(image=imgtk)
                            self.register_video_label.image = imgtk
                            no_video_shown = False
                            last_frame_time = time.time()
                    except Exception:
                        continue
                        
                else:
                    # Show no signal message after 2 seconds of no frames
                    current_time = time.time()
                    if not no_video_shown and (current_time - last_frame_time) > 4.0:
                        try:
                            if hasattr(self, 'register_video_label') and self.register_video_label is not None:
                                self.show_no_signal_message('register_video_label')
                                no_video_shown = True
                        except Exception:
                            pass
                        
                time.sleep(0.3)
                
            except Exception as e:
                print(f"Registration feed error: {e}")
                time.sleep(0.3)

    def load_registered_users(self, search_term=""):
        """Load and display registered users with enhanced search"""
        try:
            # Clear existing users
            for widget in self.users_container.winfo_children():
                widget.destroy()
                
            response = self.session.get(f"{BACKEND_URL}/registered_users")
            if response.status_code == 200:
                users = response.json()["users"]
                
                # Filter users if search term provided
                if search_term:
                    search_term = search_term.lower()
                    users = [u for u in users if 
                            search_term in u["name"].lower() or 
                            search_term in str(u["user_id"]).lower()]

                # Sort users by timestamp (most recent first)
                users.sort(key=lambda x: x["timestamp"], reverse=True)
                
                for user in users:
                    user_frame = ctk.CTkFrame(self.users_container, fg_color="#ffffff")
                    user_frame.pack(fill="x", pady=5, padx=5)
                    
                    # User details with ID
                    details_frame = ctk.CTkFrame(user_frame, fg_color="#ffffff")
                    details_frame.pack(side="left", fill="x", expand=True, padx=5, pady=5)
                    
                    id_label = ctk.CTkLabel(
                        details_frame,
                        text=f"ID: {user['user_id']}",
                        font=self.text_font,
                        anchor="w"
                    )
                    id_label.pack(fill="x")
                    
                    name_label = ctk.CTkLabel(
                        details_frame,
                        text=f"Name: {user['name']}",
                        font=self.text_font,
                        anchor="w"
                    )
                    name_label.pack(fill="x")
                    
                    date_label = ctk.CTkLabel(
                        details_frame,
                        text=f"Registered: {datetime.fromtimestamp(user['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}",
                        font=self.small_font,
                        text_color=self.text_color
                    )
                    date_label.pack(fill="x")
                    
                    # Actions frame
                    actions_frame = ctk.CTkFrame(user_frame, fg_color="#ffffff")
                    actions_frame.pack(side="right", padx=5, pady=5)
                    
                    delete_btn = ctk.CTkButton(
                        actions_frame,
                        text="Delete",
                        command=lambda id=user['face_id']: self.delete_user(id),
                        fg_color="red",
                        hover_color="darkred",
                        width=80
                    )
                    delete_btn.pack(side="right", padx=2)
                    
        except Exception as e:
            print(f"Error loading registered users: {e}")

    def delete_user(self, face_id):
        """Delete a registered user"""
        try:
            response = self.session.delete(f"{BACKEND_URL}/delete_face/{face_id}")
            if response.status_code == 200:
                self.load_registered_users()  # Refresh the list
        except Exception as e:
            print(f"Error deleting user: {e}")

    def show_settings(self):
        self.clear_screen()
        self.queue_active = False
        self.display_active = False
        
        # Main container
        settings_container = ctk.CTkFrame(self.content_area, fg_color=self.bg_color)
        settings_container.pack(fill="both", expand=True, padx=40, pady=30)
        
        # Header
        header = ctk.CTkLabel(
            settings_container,
            text="Settings & User Management",
            font=("SF Pro Display", 24, "bold"),
            text_color=self.text_color
        )
        header.pack(pady=(0, 30))
        
        # User Management Section
        user_frame = ctk.CTkFrame(settings_container, fg_color="#ffffff")
        user_frame.pack(fill="x", pady=20)
        
        # Section Header
        ctk.CTkLabel(
            user_frame,
            text="User Management",
            font=("SF Pro Display", 18, "bold"),
            text_color=self.text_color
        ).pack(pady=10, padx=20)
        
        # Form Container
        form_frame = ctk.CTkFrame(user_frame, fg_color="transparent")
        form_frame.pack(pady=10, padx=20, fill="x")
        
        # Username
        username_entry = ctk.CTkEntry(
            form_frame,
            placeholder_text="Username",
            width=250,
            height=35
        )
        username_entry.pack(side="left", padx=5)
        
        # Password
        password_entry = ctk.CTkEntry(
            form_frame,
            placeholder_text="Password",
            show="•",
            width=250,
            height=35
        )
        password_entry.pack(side="left", padx=5)
        
        # Role Selection
        role_var = ctk.StringVar(value="user")
        role_menu = ctk.CTkOptionMenu(
            form_frame,
            values=["admin", "user"],
            variable=role_var,
            width=150,
            height=35
        )
        role_menu.pack(side="left", padx=5)
        
        # Add User Button
        add_btn = ctk.CTkButton(
            form_frame,
            text="Add User",
            width=150,
            height=35,
            command=lambda: self.add_user(
                username_entry.get(),
                password_entry.get(),
                role_var.get()
            )
        )
        add_btn.pack(side="left", padx=5)
        
        # Status Label
        self.status_label = ctk.CTkLabel(
            user_frame,
            text="",
            text_color="green"
        )
        self.status_label.pack(pady=10)
        
        # Users List
        users_frame = ctk.CTkFrame(user_frame, fg_color="#f5f5f7")
        users_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Users List Header
        headers_frame = ctk.CTkFrame(users_frame, fg_color="transparent")
        headers_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(headers_frame, text="Username", width=200).pack(side="left")
        ctk.CTkLabel(headers_frame, text="Role", width=150).pack(side="left")
        ctk.CTkLabel(headers_frame, text="Actions", width=100).pack(side="left")
        
        # Scrollable Users Container
        self.users_container = ctk.CTkScrollableFrame(
            users_frame,
            fg_color="transparent"
        )
        self.users_container.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Load existing users
        self.load_app_users()

    def add_user(self, username, password, role):
        """Add a new app user"""
        if not username or not password:
            self.status_label.configure(
                text="Please fill all fields",
                text_color="red"
            )
            return
            
        try:
            response = self.session.post(
                f"{BACKEND_URL}/create_user",
                json={
                    "username": username,
                    "password": password,
                    "role": role
                }
            )
            
            if response.status_code == 200:
                self.status_label.configure(
                    text=f"User {username} created successfully",
                    text_color="green"
                )
                self.load_app_users()  # Refresh list
            else:
                self.status_label.configure(
                    text=response.json().get("detail", "Failed to create user"),
                    text_color="red"
                )
        except Exception as e:
            self.status_label.configure(
                text=f"Error: {str(e)}",
                text_color="red"
            )

    def load_app_users(self):
        """Load and display app users"""
        try:
            response = self.session.get(f"{BACKEND_URL}/users")
            if response.status_code == 200:
                users = response.json()["users"]
                
                # Clear existing entries
                for widget in self.users_container.winfo_children():
                    widget.destroy()
                
                # Add user entries
                for user in users:
                    user_frame = ctk.CTkFrame(
                        self.users_container,
                        fg_color="transparent"
                    )
                    user_frame.pack(fill="x", pady=2)
                    
                    ctk.CTkLabel(
                        user_frame,
                        text=user["username"],
                        width=200
                    ).pack(side="left")
                    
                    ctk.CTkLabel(
                        user_frame,
                        text=user["role"],
                        width=150
                    ).pack(side="left")
                    
                    if user["username"] != "admin":  # Prevent deleting admin
                        ctk.CTkButton(
                            user_frame,
                            text="Delete",
                            width=80,
                            fg_color="red",
                            hover_color="darkred",
                            command=lambda u=user["username"]: self.delete_app_user(u)
                        ).pack(side="left")
        except Exception as e:
            print(f"Error loading users: {e}")

    def delete_app_user(self, username):
        """Delete an app user"""
        try:
            response = self.session.delete(f"{BACKEND_URL}/users/{username}")
            if response.status_code == 200:
                self.status_label.configure(
                    text=f"User {username} deleted successfully",
                    text_color="green"
                )
                self.load_app_users()  # Refresh list
            else:
                self.status_label.configure(
                    text="Failed to delete user",
                    text_color="red"
                )
        except Exception as e:
            self.status_label.configure(
                text=f"Error: {str(e)}",
                text_color="red"
            )

    def update_live_feed(self):
        """Update live camera feed"""
        no_frames_shown = False
        while self.thread_running['live']:
            try:
                # Check if any video labels exist
                if not any(hasattr(self, f'video_label{i}') for i in range(1, 4)):
                    time.sleep(0.1)
                    continue
                    
                response = self.session.get(f"{BACKEND_URL}/get_latest_frame")
                if response.status_code == 200:
                    frame_bytes = response.content
                    # Decode frame in BGR format first
                    frame = cv2.imdecode(
                        np.frombuffer(frame_bytes, np.uint8),
                        cv2.IMREAD_COLOR
                    )
                    # Convert BGR to RGB for display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # For recognition pipeline
                    if self.queue_active:
                        try:
                            self.frame_queue.put_nowait(frame)
                        except queue.Full:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(frame)
                    
                    # For display feed and live recognition
                    if self.display_active:
                        frame_with_faces, faces_detected = self.detect_faces(frame_rgb)
                        display_image = Image.fromarray(frame_with_faces)
                        imgtk = ctk.CTkImage(
                            light_image=display_image, 
                            dark_image=display_image, 
                            size=(800, 600)
                        )
                        
                        if hasattr(self, 'video_label1'):
                            self.video_label1.configure(image=imgtk)
                            self.video_label1.image = imgtk

                        # Handle live recognition
                        if faces_detected and hasattr(self, 'live_recognition_label'):
                            try:
                                # Process image for recognition
                                img_byte_arr = BytesIO()
                                Image.fromarray(frame_rgb).save(img_byte_arr, format='PNG')
                                files = {'file': ('image.png', img_byte_arr.getvalue(), 'image/png')}
                                
                                recognition_response = self.session.post(
                                    f"{BACKEND_URL}/recognize_face",
                                    files=files
                                )
                                
                                if recognition_response.status_code == 200:
                                    result = recognition_response.json()
                                    name = result["name"]
                                    confidence = result["confidence"] * 100
                                    self.live_recognition_label.configure(
                                        text=f"Recognized: {name}\nConfidence: {confidence:.2f}%",
                                        text_color="green" if confidence > 50 else "orange"
                                    )
                                else:
                                    self.live_recognition_label.configure(
                                        text="Unknown Person", 
                                        text_color="red"
                                    )
                            except Exception as e:
                                print(f"Live recognition error: {e}")
                        elif hasattr(self, 'live_recognition_label'):
                            self.live_recognition_label.configure(
                                text="Awaiting face detection...",
                                text_color=self.text_color
                            )
                    no_frames_shown = False
                    time.sleep(0.1)  # Rate limiting
                else:
                    if not no_frames_shown:
                        try:
                            for i in range(1, 4):
                                label_name = f'video_label{i}'
                                if hasattr(self, label_name):
                                    self.show_no_signal_message(label_name)
                            if hasattr(self, 'live_recognition_label'):
                                self.live_recognition_label.configure(
                                    text="Camera disconnected",
                                    text_color="red"
                                )
                            no_frames_shown = True
                        except Exception:
                            pass
                    time.sleep(0.1)
            except Exception:
                if not no_frames_shown:
                    try:
                        for i in range(1, 4):
                            label_name = f'video_label{i}'
                            if hasattr(self, label_name):
                                self.show_no_signal_message(label_name)
                        if hasattr(self, 'live_recognition_label'):
                            self.live_recognition_label.configure(
                                text="Camera disconnected",
                                text_color="red"
                            )
                        no_frames_shown = True
                    except Exception:
                        pass
                time.sleep(0.1)

    def capture_face_embedding(self):
        name = self.name_entry.get().strip()
        user_id = self.id_entry.get().strip()
        
        if not user_id:
            self.feedback_label.configure(text="Please enter an ID", text_color="red")
            return
        if not name:
            self.feedback_label.configure(text="Please enter a name", text_color="red")
            return
        if not hasattr(self, 'current_frame'):
            self.feedback_label.configure(text="No frame available", text_color="red")
            return

        # Send to backend for embedding
        try:
            img_byte_arr = BytesIO()
            Image.fromarray(self.current_frame).save(img_byte_arr, format='PNG')
            files = {'file': ('image.png', img_byte_arr.getvalue(), 'image/png')}
            data = {'name': name, 'user_id': user_id}
            response = self.session.post(f"{BACKEND_URL}/register_face", files=files, data=data)
            
            if response.status_code == 200:
                self.feedback_label.configure(text=f"Face registered for {name} (ID: {user_id})", text_color="green")
                self.id_entry.delete(0, 'end')
                self.name_entry.delete(0, 'end')
                self.load_registered_users()
            else:
                error_msg = response.json().get('detail', 'Registration failed')
                self.feedback_label.configure(text=error_msg, text_color="red")
        except Exception as e:
            self.feedback_label.configure(text=f"Error: {str(e)}", text_color="red")

    def exit_app(self):
        """Modified to remove camera release"""
        for key in self.thread_running:
            self.thread_running[key] = False
            
        for thread in self.threads.values():
            if thread:
                thread.join(timeout=1.0)
                
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":
    root = ctk.CTk()
    app = CameraApp(root)
    root.mainloop()
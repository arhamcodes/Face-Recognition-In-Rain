import numpy as np
import cv2
import customtkinter as ctk
import torch
import threading
import os
from torchvision import transforms
from PIL import Image, ImageTk, ImageEnhance, ImageDraw
from pathlib import Path
import time

# Add icons path constant
ICONS_PATH = Path("assets/icons")  # Create this folder and add your icons

# Add after ICONS_PATH constant
def init_assets():
    """Initialize required directories and assets"""
    if not ICONS_PATH.exists():
        ICONS_PATH.mkdir(parents=True, exist_ok=True)
        print(f"Created icons directory at {ICONS_PATH}")
    
    # Check if required icons exist
    required_icons = ['live.png', 'face.png', 'add.png', 'chart.png', 'settings.png']
    missing_icons = []
    for icon in required_icons:
        if not (ICONS_PATH / icon).exists():
            missing_icons.append(icon)
    
    if missing_icons:
        print(f"Warning: Missing icons: {', '.join(missing_icons)}")
        print(f"Please add icons to: {ICONS_PATH}")

# Load the pre-trained TorchScript model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = torch.jit.load("models/generator_model.pt", map_location=device)
generator.eval()

# Dummy embedding model (to be replaced with actual face encoder)
def get_face_embedding(image):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    image = cv2.resize(image, (128, 128))
    return np.random.rand(128).astype('float32')

# Image Preprocessing Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class CameraApp:
    def __init__(self, root):
        init_assets()  # Initialize assets before setting up UI
        self.root = root
        self.root.title("Face Recognition System")
        self.root.state("zoomed")
        self.root.minsize(1400, 768)  # Set minimum window size
        
        # Add thread state tracking
        self.threads_started = False
        self.register_running = True  # Add register feed state
        self.register_thread = None
        
        # Configure theme
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        
        # Configure fonts
        self.title_font = ("SF Pro Display", 20, "bold")
        self.text_font = ("SF Pro Text", 12)
        self.small_font = ("SF Pro Text", 10)
        
        # Configure colors
        self.bg_color = "#ffffff"
        self.accent_color = "#0066cc"
        self.text_color = "#333333"
        self.secondary_color = "#f5f5f7"
        
        self.setup_ui()
        self.init_camera()  # Initialize camera after UI setup
        
        # Start register feed thread
        self.register_thread = threading.Thread(target=self.update_register_feed, daemon=True)
        self.register_thread.start()
        
    def init_camera(self):
        """Initialize camera and video label attributes"""
        self.cap = cv2.VideoCapture(0)
        self.running = False  # Start as False
        
        # Initialize video labels
        self.video_label1 = None
        self.video_label2 = None 
        self.video_label3 = None
        self.register_video_label = None
        self.current_frame = None
        
        # Initialize but don't start threads
        self.live_thread = None
        self.rain_thread = None
        self.process_thread = None

    def setup_ui(self):
        # Main container with subtle padding
        self.main_container = ctk.CTkFrame(
            self.root,
            fg_color=self.bg_color,
            corner_radius=0
        )
        self.main_container.pack(fill="both", expand=True, padx=20, pady=20)

        # Sidebar with icons
        self.sidebar = ctk.CTkFrame(
            self.main_container,
            fg_color=self.secondary_color,
            width=80,
            corner_radius=0
        )
        self.sidebar.pack(side="left", fill="y", padx=(0, 20))

        # Create navigation buttons with icons
        self.nav_buttons = [
            ("Live", "live.png", self.show_live_feed),
            ("Recognize", "face.png", self.show_recognition),
            ("Register", "add.png", self.show_register_face),
            ("Analytics", "chart.png", self.show_analytics),
            ("Settings", "settings.png", self.show_settings)
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

    def create_video_frame(self, text):
        frame = ctk.CTkFrame(
            self.video_container,
            fg_color=self.secondary_color,
            corner_radius=0,
            width=400,  # Increased size
            height=400  # Increased size
        )
        frame.pack_propagate(False)  # Prevent frame from shrinking
        
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
        
        # Stop existing threads if running
        self.running = False
        if self.threads_started:
            if self.live_thread: self.live_thread.join(timeout=1.0)
            if self.rain_thread: self.rain_thread.join(timeout=1.0)
            if self.process_thread: self.process_thread.join(timeout=1.0)
        
        # Header
        header = ctk.CTkLabel(
            self.content_area,
            text="Live Camera Feed",
            font=self.title_font,
            text_color=self.text_color
        )
        header.pack(pady=(0, 20))
        
        # Video container with grid layout
        self.video_container = ctk.CTkFrame(
            self.content_area,
            fg_color=self.bg_color,
            corner_radius=0
        )
        self.video_container.pack(expand=True)
        
        # Create and position video frames
        frames = [
            ("Live Feed", "video_label1"),
            ("Original", "video_label2"),
            ("Processed", "video_label3")
        ]
        
        for i, (title, label_name) in enumerate(frames):
            frame = self.create_video_frame(title)
            frame.grid(row=0, column=i, padx=10)
            
            # Create and store video label
            label = ctk.CTkLabel(frame, text="")
            label.pack(expand=True, fill="both")
            setattr(self, label_name, label)
        
        # Create new threads
        self.running = True
        self.live_thread = threading.Thread(target=self.update_live_feed, daemon=True)
        self.rain_thread = threading.Thread(target=self.update_rain_output, daemon=True)
        self.process_thread = threading.Thread(target=self.update_processed_output, daemon=True)
        
        # Start the threads
        self.live_thread.start()
        self.rain_thread.start()
        self.process_thread.start()
        self.threads_started = True

    def clear_screen(self):
        # Stop video threads before clearing
        self.running = False
        if self.threads_started:
            if self.live_thread: self.live_thread.join(timeout=1.0)
            if self.rain_thread: self.rain_thread.join(timeout=1.0)
            if self.process_thread: self.process_thread.join(timeout=1.0)
            self.threads_started = False
        
        for widget in self.content_area.winfo_children():
            widget.destroy()

    def create_video_label(self, frame):
        label = ctk.CTkLabel(frame, text="")
        label.pack()
        return label

    def show_recognition(self):
        self.clear_screen()
        label = ctk.CTkLabel(self.content_area, text="Face Recognition & Confidence", font=self.title_font)
        label.pack(pady=50)

    def show_register_face(self):
        self.clear_screen()

        self.register_frame = ctk.CTkFrame(self.content_area)
        self.register_frame.pack(pady=20)

        name_label = ctk.CTkLabel(self.register_frame, text="Full Name:", font=self.text_font)
        name_label.grid(row=0, column=0, padx=10, pady=10)
        self.name_entry = ctk.CTkEntry(self.register_frame, width=300)
        self.name_entry.grid(row=0, column=1, padx=10, pady=10)

        capture_button = ctk.CTkButton(self.register_frame, text="Register Face", command=self.capture_face_embedding, width=300)
        capture_button.grid(row=1, column=0, columnspan=2, pady=10)

        self.feedback_label = ctk.CTkLabel(self.content_area, text="", font=self.text_font)
        self.feedback_label.pack(pady=10)

        self.live_register_frame = ctk.CTkFrame(self.content_area)
        self.live_register_frame.pack()
        self.register_video_label = ctk.CTkLabel(self.live_register_frame, text="")
        self.register_video_label.pack()

    def show_analytics(self):
        self.clear_screen()
        label = ctk.CTkLabel(self.content_area, text="Recognition Analytics & Logs", font=self.title_font)
        label.pack(pady=50)

    def show_settings(self):
        self.clear_screen()
        label = ctk.CTkLabel(self.content_area, text="Settings & Model Management", font=self.title_font)
        label.pack(pady=50)

    def apply_model(self, frame):
        img = Image.fromarray(frame)
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = generator(img)
            output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        processed_frame = (output * 255).astype("uint8")
        return processed_frame

    def update_register_feed(self):
        while self.register_running:  # Use separate running flag
            try:
                ret, frame = self.cap.read()
                if not ret: continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame).resize((256, 256), Image.Resampling.LANCZOS)
                imgtk = ctk.CTkImage(light_image=image, dark_image=image, size=(256, 256))
                if hasattr(self, 'register_video_label'):
                    self.register_video_label.configure(image=imgtk)
                    self.register_video_label.image = imgtk
                self.current_frame = frame
                time.sleep(0.03)  # Add small delay to reduce CPU usage
            except Exception as e:
                # print(f"Error in register feed: {e}")
                time.sleep(0.1)

    def capture_face_embedding(self):
        name = self.name_entry.get().strip()
        if not name:
            self.feedback_label.configure(text="Please enter a name", text_color="red")
            return
        if not hasattr(self, 'current_frame'):
            self.feedback_label.configure(text="No frame available", text_color="red")
            return

        embedding = get_face_embedding(self.current_frame)
        # Save embedding logic here
        self.feedback_label.configure(text=f"Face embedding saved for {name}", text_color="green")

    def exit_app(self):
        self.register_running = False  # Stop register thread
        self.running = False
        if self.register_thread:
            self.register_thread.join(timeout=1.0)
        self.cap.release()
        self.root.quit()
        self.root.destroy()

    def update_live_feed(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret: continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb).resize((384, 384), Image.Resampling.LANCZOS)
                imgtk = ctk.CTkImage(light_image=image, dark_image=image, size=(384, 384))
                
                if hasattr(self, 'video_label1') and self.video_label1 is not None:
                    self.video_label1.configure(image=imgtk)
                    self.video_label1.image = imgtk  # Keep reference!
            except Exception as e:
                print(f"Error in live feed: {e}")
                time.sleep(0.1)

    def update_rain_output(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret: continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb).resize((256, 256), Image.Resampling.LANCZOS)
            imgtk = ctk.CTkImage(light_image=image, dark_image=image, size=(256, 256))

            if hasattr(self, 'video_label2'):
                self.video_label2.configure(image=imgtk)
                self.video_label2.image = imgtk

    def update_processed_output(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret: continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = self.apply_model(frame_rgb)
            processed_image = Image.fromarray(processed_frame)
            processed_imgtk = ctk.CTkImage(light_image=processed_image, dark_image=processed_image, size=(256, 256))

            if hasattr(self, 'video_label3'):
                self.video_label3.configure(image=processed_imgtk)
                self.video_label3.image = processed_imgtk

if __name__ == "__main__":
    root = ctk.CTk()
    app = CameraApp(root)
    root.mainloop()

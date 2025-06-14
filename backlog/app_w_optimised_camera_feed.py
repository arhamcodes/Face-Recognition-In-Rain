import numpy as np
import cv2
import customtkinter as ctk
import torch
import threading
import time
from torchvision import transforms
from PIL import Image, ImageTk

# Load the pre-trained TorchScript model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = torch.jit.load("generator_model.pt", map_location=device)
generator.eval().to(device)

# Image Preprocessing Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI-Powered Camera App")
        self.root.state("zoomed")

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")  

        # Video Capture
        self.cap = cv2.VideoCapture(0)

        # Reduce latency by lowering camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Main Frame
        self.main_frame = ctk.CTkFrame(self.root, fg_color="#F3E5F5")
        self.main_frame.pack(expand=True, fill="both")

        # Video Container
        self.video_container = ctk.CTkFrame(self.main_frame)
        self.video_container.pack(expand=True, pady=20)

        # Create Video Frames
        self.frame1 = self.create_video_frame("Live Feed")
        self.frame2 = self.create_video_frame("Processed Output")

        # Create Labels for Video Display
        self.video_label1 = self.create_video_label(self.frame1)
        self.video_label2 = self.create_video_label(self.frame2)

        self.frame1.grid(row=0, column=0, padx=30, pady=20)
        self.frame2.grid(row=0, column=1, padx=30, pady=20)

        self.video_container.grid_columnconfigure((0, 1), weight=1)

        # Start Threads
        self.running = True
        self.processed_frame = None
        self.video_thread = threading.Thread(target=self.update_video_feed, daemon=True)
        self.model_thread = threading.Thread(target=self.model_processing, daemon=True)
        
        self.video_thread.start()
        self.model_thread.start()

    def create_video_frame(self, text):
        frame = ctk.CTkFrame(self.video_container, width=256, height=256, fg_color="#3a3f58", corner_radius=15)
        ctk.CTkLabel(frame, text=text, font=("SF Pro Display", 14, "bold")).pack(pady=5)
        return frame

    def create_video_label(self, frame):
        label = ctk.CTkLabel(frame, text="")
        label.pack()
        return label

    def apply_model(self, frame):
        """ Pass the frame through the Generator model and return the processed output """
        img = cv2.resize(frame, (256, 256))  # Use OpenCV for faster resizing
        img = transform(img).unsqueeze(0).to(device, non_blocking=True)  # Convert to Tensor and move to GPU

        with torch.no_grad():
            output = generator(img).squeeze(0).permute(1, 2, 0).cpu().numpy()

        return (output * 255).astype("uint8")  # Convert to uint8 format

    def update_video_feed(self):
        """ Continuously fetch frames and update the UI """
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            # Display live feed
            image = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.video_label1.imgtk = image
            self.video_label1.configure(image=image)

            # Display processed feed (only update if available)
            if self.processed_frame is not None:
                processed_imgtk = ImageTk.PhotoImage(image=Image.fromarray(self.processed_frame))
                self.video_label2.imgtk = processed_imgtk
                self.video_label2.configure(image=processed_imgtk)

            time.sleep(0.01)  # Control refresh rate (~100 FPS)

    def model_processing(self):
        """ Runs the model on the latest frame asynchronously to avoid blocking the UI """
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.processed_frame = self.apply_model(frame)

    def exit_app(self):
        self.running = False
        self.cap.release()
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":
    root = ctk.CTk()
    app = CameraApp(root)
    root.mainloop()

# import cv2
# import customtkinter as ctk
# from PIL import Image, ImageTk
# import threading

# class CameraApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("AI-Powered Camera App")
#         self.root.state("zoomed")  # Windows full-screen mode
#         ctk.set_appearance_mode("light")  # Light mode UI
#         ctk.set_default_color_theme("blue")
        
#         # Apply gradient background
#         self.root.configure(bg="#EDE7F6")
        
#         # Main Container
#         self.main_frame = ctk.CTkFrame(self.root, fg_color="#F3E5F5")
#         self.main_frame.pack(expand=True, fill="both")
        
#         # Video Capture
#         self.cap = cv2.VideoCapture(0)
        
#         # Navigation Pane
#         self.nav_pane = ctk.CTkFrame(self.root, width=250, fg_color="#673AB7")
#         self.nav_pane.pack(side="left", fill="y")
        
#         ctk.CTkLabel(self.nav_pane, text="Navigation", font=("Inter", 20, "bold"), text_color="#FFFFFF").pack(pady=20)
#         self.home_btn = ctk.CTkButton(self.nav_pane, text="Home", font=("Inter", 14), corner_radius=10, fg_color="#9575CD")
#         self.settings_btn = ctk.CTkButton(self.nav_pane, text="Settings", font=("Inter", 14), corner_radius=10, fg_color="#9575CD")
#         self.exit_button = ctk.CTkButton(self.nav_pane, text="Exit", font=("Inter", 14), command=self.exit_app, fg_color="#D32F2F", corner_radius=10)
        
#         self.home_btn.pack(pady=10, fill="x", padx=20)
#         self.settings_btn.pack(pady=10, fill="x", padx=20)
#         self.exit_button.pack(pady=10, fill="x", padx=20)
        
#         # Frames for Video Display with gradient background
#         self.video_container = ctk.CTkFrame(self.main_frame, fg_color="#E1BEE7")
#         self.video_container.pack(expand=True, pady=20)
        
#         self.frame1 = self.create_video_frame("Live Feed")
#         self.frame2 = self.create_video_frame("Processed Output 1")
#         self.frame3 = self.create_video_frame("Processed Output 2")
        
#         self.video_label1 = self.create_video_label(self.frame1)
#         self.video_label2 = self.create_video_label(self.frame2)
#         self.video_label3 = self.create_video_label(self.frame3)
        
#         # Arrange frames in a centered grid
#         self.frame1.grid(row=0, column=0, padx=30, pady=20)
#         self.frame2.grid(row=0, column=1, padx=30, pady=20)
#         self.frame3.grid(row=0, column=2, padx=30, pady=20)
        
#         self.video_container.grid_columnconfigure(0, weight=1)
#         self.video_container.grid_columnconfigure(1, weight=1)
#         self.video_container.grid_columnconfigure(2, weight=1)
        
#         # Start Video Thread
#         self.running = True
#         self.video_thread = threading.Thread(target=self.update_video_feed)
#         self.video_thread.start()
    
#     def create_video_frame(self, text):
#         frame = ctk.CTkFrame(self.video_container, width=256, height=256, fg_color="#9575CD", corner_radius=15)
#         ctk.CTkLabel(frame, text=text, font=("Inter", 16, "bold"), text_color="#FFFFFF").pack(pady=5)
#         return frame
    
#     def create_video_label(self, frame):
#         label = ctk.CTkLabel(frame, text="", fg_color="#D1C4E9")
#         label.pack()
#         return label
    
#     def update_video_feed(self):
#         while self.running:
#             ret, frame = self.cap.read()
#             if not ret:
#                 break
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             image = Image.fromarray(frame)
#             image = image.resize((256, 256), Image.Resampling.LANCZOS)
#             imgtk = ImageTk.PhotoImage(image=image)
            
#             self.video_label1.imgtk = imgtk
#             self.video_label1.configure(image=imgtk)
            
#             self.video_label2.imgtk = imgtk  # Placeholder, replace with ML output
#             self.video_label2.configure(image=imgtk)
            
#             self.video_label3.imgtk = imgtk  # Placeholder, replace with ML output
#             self.video_label3.configure(image=imgtk)
            
#             self.root.update()
    
#     def exit_app(self):
#         self.running = False
#         self.cap.release()
#         self.root.quit()
#         self.root.destroy()

# if __name__ == "__main__":
#     root = ctk.CTk()
#     app = CameraApp(root)
#     root.mainloop()

# import numpy as np
# import cv2
# import customtkinter as ctk
# import torch
# import threading
# from torchvision import transforms
# from PIL import Image, ImageTk
# from derain import IDGAN  # Import your GAN class
# from derain import DataLoader

# gan = IDGAN()  # Initialize the model
# # gan.load_checkpoint("checkpoint_434.pth")  # Load model weights
# checkpoint_path = "checkpoint_434.pth"
# checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# # # Load the PyTorch Model
# # model = torch.load("checkpoint_434.pth", map_location=torch.device("cpu"))
# # model.eval()

# # Image Preprocessing Transform
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])

# class CameraApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("AI-Powered Camera App")
#         self.root.state("zoomed")

#         ctk.set_appearance_mode("light")
#         # ctk.set_default_color_theme("purple_theme.json")  # Your custom theme
#         ctk.set_default_color_theme("blue")  

#         # Video Capture
#         self.cap = cv2.VideoCapture(0)

#         # Main Frame
#         self.main_frame = ctk.CTkFrame(self.root, fg_color="#F3E5F5")
#         self.main_frame.pack(expand=True, fill="both")

#         # Video Container
#         self.video_container = ctk.CTkFrame(self.main_frame)
#         self.video_container.pack(expand=True, pady=20)

#         # Create Video Frames
#         self.frame1 = self.create_video_frame("Live Feed")
#         self.frame2 = self.create_video_frame("Processed Output 1")
#         self.frame3 = self.create_video_frame("Processed Output 2")

#         # Create Labels for Video Display
#         self.video_label1 = self.create_video_label(self.frame1)
#         self.video_label2 = self.create_video_label(self.frame2)
#         self.video_label3 = self.create_video_label(self.frame3)

#         self.frame1.grid(row=0, column=0, padx=30, pady=20)
#         self.frame2.grid(row=0, column=1, padx=30, pady=20)
#         self.frame3.grid(row=0, column=2, padx=30, pady=20)

#         self.video_container.grid_columnconfigure((0, 1, 2), weight=1)

#         # Start Video Thread
#         self.running = True
#         self.video_thread = threading.Thread(target=self.update_video_feed)
#         self.video_thread.start()

#     def create_video_frame(self, text):
#         frame = ctk.CTkFrame(self.video_container, width=256, height=256, fg_color="#3a3f58", corner_radius=15)
#         ctk.CTkLabel(frame, text=text, font=("SF Pro Display", 14, "bold")).pack(pady=5)
#         return frame

#     def create_video_label(self, frame):
#         label = ctk.CTkLabel(frame, text="")
#         label.pack()
#         return label

#     # def apply_model(self, frame):
#     #     """ Pass the frame through the ML model and return processed output """
#     #     img = Image.fromarray(frame)  # Convert OpenCV image to PIL
#     #     img = transform(img).unsqueeze(0)  # Apply transformations and add batch dimension

#     #     with torch.no_grad():
#     #         output = model(img)  # Run model inference
        
#     #     # Process the output (convert it to an image if needed)
#     #     processed_frame = output.squeeze(0).permute(1, 2, 0).numpy()  # Convert back to NumPy
#     #     processed_frame = (processed_frame * 255).astype("uint8")  # Convert to uint8 format
#     #     return processed_frame

#     def apply_model(self, frame):
#         """ Pass the frame through the GAN model and return the processed output """
#         img = Image.fromarray(frame)  # Convert OpenCV image to PIL
#         img = img.resize((256, 256), Image.Resampling.LANCZOS)  # Resize

#         # Convert to dataset format
#         test_loader = DataLoader(dataset_name='rain', img_res=(256, 256), single_image=img)

#         # Run inference
#         output = gan.test(test_loader)

#         # Convert back to OpenCV format
#         processed_frame = np.array(output)  # Convert to NumPy
#         processed_frame = (processed_frame * 255).astype("uint8")  # Convert to uint8
#         return processed_frame
        
    
#     def update_video_feed(self):
#         while self.running:
#             ret, frame = self.cap.read()
#             if not ret:
#                 break

#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
#             image = Image.fromarray(frame).resize((256, 256), Image.Resampling.LANCZOS)
#             imgtk = ImageTk.PhotoImage(image=image)

#             # Display the original frame in Live Feed
#             self.video_label1.imgtk = imgtk
#             self.video_label1.configure(image=imgtk)

#             # Placeholder: Show same frame in Processed Output 1
#             self.video_label2.imgtk = imgtk
#             self.video_label2.configure(image=imgtk)

#             # Run the frame through the ML model and display in Processed Output 2
#             processed_frame = self.apply_model(frame)
#             processed_image = Image.fromarray(processed_frame)
#             processed_imgtk = ImageTk.PhotoImage(image=processed_image)

#             self.video_label3.imgtk = processed_imgtk
#             self.video_label3.configure(image=processed_imgtk)

#             self.root.update()

#     def exit_app(self):
#         self.running = False
#         self.cap.release()
#         self.root.quit()
#         self.root.destroy()

# if __name__ == "__main__":
#     root = ctk.CTk()
#     app = CameraApp(root)
#     root.mainloop()


# import numpy as np
# import cv2
# import customtkinter as ctk
# import torch
# import threading
# from torchvision import transforms
# from PIL import Image, ImageTk

# # Load the pre-trained TorchScript model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# generator = torch.jit.load("generator_model.pt", map_location=device)
# generator.eval()

# # Image Preprocessing Transform
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])

# class CameraApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("AI-Powered Camera App")
#         self.root.state("zoomed")

#         ctk.set_appearance_mode("light")
#         ctk.set_default_color_theme("blue")  

#         # Video Capture
#         self.cap = cv2.VideoCapture(0)

#         # Main Frame
#         self.main_frame = ctk.CTkFrame(self.root, fg_color="#F3E5F5")
#         self.main_frame.pack(expand=True, fill="both")

#         # Video Container
#         self.video_container = ctk.CTkFrame(self.main_frame)
#         self.video_container.pack(expand=True, pady=20)

#         # Create Video Frames
#         self.frame1 = self.create_video_frame("Live Feed")
#         self.frame2 = self.create_video_frame("Processed Output 1")
#         self.frame3 = self.create_video_frame("Processed Output 2")

#         # Create Labels for Video Display
#         self.video_label1 = self.create_video_label(self.frame1)
#         self.video_label2 = self.create_video_label(self.frame2)
#         self.video_label3 = self.create_video_label(self.frame3)

#         self.frame1.grid(row=0, column=0, padx=30, pady=20)
#         self.frame2.grid(row=0, column=1, padx=30, pady=20)
#         self.frame3.grid(row=0, column=2, padx=30, pady=20)

#         self.video_container.grid_columnconfigure((0, 1, 2), weight=1)

#         # Start Video Thread
#         self.running = True
#         self.video_thread = threading.Thread(target=self.update_video_feed)
#         self.video_thread.start()

#     def create_video_frame(self, text):
#         frame = ctk.CTkFrame(self.video_container, width=256, height=256, fg_color="#3a3f58", corner_radius=15)
#         ctk.CTkLabel(frame, text=text, font=("SF Pro Display", 14, "bold")).pack(pady=5)
#         return frame

#     def create_video_label(self, frame):
#         label = ctk.CTkLabel(frame, text="")
#         label.pack()
#         return label

#     def apply_model(self, frame):
#         """ Pass the frame through the Generator model and return the processed output """
#         img = Image.fromarray(frame)  # Convert OpenCV image to PIL
#         img = transform(img).unsqueeze(0).to(device)  # Apply transformations and move to device

#         with torch.no_grad():
#             output = generator(img)  # Run inference
#             output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Convert to NumPy

#         # Convert back to OpenCV format
#         processed_frame = (output * 255).astype("uint8")  # Convert to uint8 format
#         return processed_frame

#     def update_video_feed(self):
#         while self.running:
#             ret, frame = self.cap.read()
#             if not ret:
#                 break

#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
#             image = Image.fromarray(frame).resize((256, 256), Image.Resampling.LANCZOS)
#             imgtk = ImageTk.PhotoImage(image=image)

#             # Display the original frame in Live Feed
#             self.video_label1.imgtk = imgtk
#             self.video_label1.configure(image=imgtk)

#             # Placeholder: Show same frame in Processed Output 1
#             self.video_label2.imgtk = imgtk
#             self.video_label2.configure(image=imgtk)

#             # Run the frame through the ML model and display in Processed Output 2
#             processed_frame = self.apply_model(frame)
#             processed_image = Image.fromarray(processed_frame)
#             processed_imgtk = ImageTk.PhotoImage(image=processed_image)

#             self.video_label3.imgtk = processed_imgtk
#             self.video_label3.configure(image=processed_imgtk)

#             self.root.update()

#     def exit_app(self):
#         self.running = False
#         self.cap.release()
#         self.root.quit()
#         self.root.destroy()

# if __name__ == "__main__":
#     root = ctk.CTk()
#     app = CameraApp(root)
#     root.mainloop()


# import numpy as np
# import cv2
# import customtkinter as ctk
# import torch
# import threading
# import os
# from torchvision import transforms
# from PIL import Image, ImageTk
# import chromadb
# from chromadb.config import Settings

# # Load the pre-trained TorchScript model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# generator = torch.jit.load("generator_model.pt", map_location=device)
# generator.eval()

# # Initialize ChromaDB client and collection
# chroma_client = chromadb.Client(Settings(persist_directory="./chroma_storage", anonymized_telemetry=False))
# if "face_embeddings" not in chroma_client.list_collections():
#     collection = chroma_client.create_collection(name="face_embeddings")
# else:
#     collection = chroma_client.get_collection(name="face_embeddings")

# # Dummy embedding model (to be replaced with actual face encoder)
# def get_face_embedding(image):
#     if image.dtype != np.uint8:
#         image = (image * 255).astype(np.uint8)
#     image = cv2.resize(image, (128, 128))
#     return np.random.rand(128).astype('float32')

# # Image Preprocessing Transform
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])

# class CameraApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Face Recognition in Rain")
#         self.root.state("zoomed")

#         ctk.set_appearance_mode("light")
#         ctk.set_default_color_theme("blue")

#         self.current_screen = None
#         self.cap = cv2.VideoCapture(0)
#         self.running = True

#         # Navigation bar
#         self.navbar = ctk.CTkFrame(self.root, height=50)
#         self.navbar.pack(fill="x")
#         ctk.CTkButton(self.navbar, text="Live Feed", command=self.show_live_feed).pack(side="left", padx=10, pady=10)
#         ctk.CTkButton(self.navbar, text="Recognition", command=self.show_recognition).pack(side="left", padx=10, pady=10)
#         ctk.CTkButton(self.navbar, text="Register Face", command=self.show_register_face).pack(side="left", padx=10, pady=10)
#         ctk.CTkButton(self.navbar, text="Analytics", command=self.show_analytics).pack(side="left", padx=10, pady=10)
#         ctk.CTkButton(self.navbar, text="Settings", command=self.show_settings).pack(side="left", padx=10, pady=10)

#         self.container = ctk.CTkFrame(self.root)
#         self.container.pack(expand=True, fill="both")

#         self.show_live_feed()

#     def clear_screen(self):
#         for widget in self.container.winfo_children():
#             widget.destroy()

#     def show_live_feed(self):
#         self.clear_screen()

#         self.video_container = ctk.CTkFrame(self.container)
#         self.video_container.pack(expand=True, pady=20)

#         self.frame1 = self.create_video_frame("Live Feed")
#         self.frame2 = self.create_video_frame("No Rain Output")
#         self.frame3 = self.create_video_frame("Rain Augmented")

#         self.video_label1 = self.create_video_label(self.frame1)
#         self.video_label2 = self.create_video_label(self.frame2)
#         self.video_label3 = self.create_video_label(self.frame3)

#         self.frame1.grid(row=0, column=0, padx=30, pady=20)
#         self.frame2.grid(row=0, column=1, padx=30, pady=20)
#         self.frame3.grid(row=0, column=2, padx=30, pady=20)
#         self.video_container.grid_columnconfigure((0, 1, 2), weight=1)

#         self.running = True 
#         self.live_thread = threading.Thread(target=self.update_live_feed)
#         self.rain_thread = threading.Thread(target=self.update_rain_output)
#         self.process_thread = threading.Thread(target=self.update_processed_output)

#         self.live_thread.start()
#         self.rain_thread.start()
#         self.process_thread.start()

#     def show_recognition(self):
#         self.clear_screen()
#         label = ctk.CTkLabel(self.container, text="Face Recognition & Confidence", font=("Arial", 20))
#         label.pack(pady=50)

#     def show_register_face(self):
#         self.clear_screen()

#         self.register_frame = ctk.CTkFrame(self.container)
#         self.register_frame.pack(pady=20)

#         name_label = ctk.CTkLabel(self.register_frame, text="Full Name:")
#         name_label.grid(row=0, column=0, padx=10, pady=10)
#         self.name_entry = ctk.CTkEntry(self.register_frame, width=300)
#         self.name_entry.grid(row=0, column=1, padx=10, pady=10)

#         capture_button = ctk.CTkButton(self.register_frame, text="Register Face", command=self.capture_face_embedding)
#         capture_button.grid(row=1, column=0, columnspan=2, pady=10)

#         self.feedback_label = ctk.CTkLabel(self.container, text="")
#         self.feedback_label.pack(pady=10)

#         self.live_register_frame = ctk.CTkFrame(self.container)
#         self.live_register_frame.pack()
#         self.register_video_label = ctk.CTkLabel(self.live_register_frame, text="")
#         self.register_video_label.pack()

#         threading.Thread(target=self.update_register_feed).start()

#     def show_analytics(self):
#         self.clear_screen()
#         label = ctk.CTkLabel(self.container, text="Recognition Analytics & Logs", font=("Arial", 20))
#         label.pack(pady=50)

#     def show_settings(self):
#         self.clear_screen()
#         label = ctk.CTkLabel(self.container, text="Settings & Model Management", font=("Arial", 20))
#         label.pack(pady=50)

#     def create_video_frame(self, text):
#         frame = ctk.CTkFrame(self.video_container, width=256, height=256, fg_color="#3a3f58", corner_radius=15)
#         ctk.CTkLabel(frame, text=text, font=("SF Pro Display", 14, "bold")).pack(pady=5)
#         return frame

#     def create_video_label(self, frame):
#         label = ctk.CTkLabel(frame, text="")
#         label.pack()
#         return label

#     def apply_model(self, frame):
#         img = Image.fromarray(frame)
#         img = transform(img).unsqueeze(0).to(device)
#         with torch.no_grad():
#             output = generator(img)
#             output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
#         processed_frame = (output * 255).astype("uint8")
#         return processed_frame

#     def update_register_feed(self):
#         while self.running:
#             ret, frame = self.cap.read()
#             if not ret:
#                 continue
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             image = Image.fromarray(frame).resize((256, 256), Image.Resampling.LANCZOS)
#             imgtk = ImageTk.PhotoImage(image=image)
#             if hasattr(self, 'register_video_label'):
#                 self.register_video_label.imgtk = imgtk
#                 self.register_video_label.configure(image=imgtk)
#             self.current_frame = frame
#             self.root.update()

#     def capture_face_embedding(self):
#         name = self.name_entry.get().strip()
#         if not name:
#             self.feedback_label.configure(text="Please enter a name", text_color="red")
#             return
#         if not hasattr(self, 'current_frame'):
#             self.feedback_label.configure(text="No frame available", text_color="red")
#             return

#         embedding = get_face_embedding(self.current_frame)
#         uid = f"{name}_{len(collection.get(ids=[]).get('ids', []))}"
#         collection.add(embeddings=[embedding.tolist()], metadatas=[{"name": name}], ids=[uid])

#         self.feedback_label.configure(text=f"Face embedding saved for {name}", text_color="green")

#     def exit_app(self):
#         self.running = False
#         self.cap.release()
#         self.root.quit()
#         self.root.destroy()

#     def update_live_feed(self):
#         while self.running:
#             ret, frame = self.cap.read()
#             if not ret: continue
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             image = Image.fromarray(frame_rgb).resize((256, 256), Image.Resampling.LANCZOS)
#             imgtk = ImageTk.PhotoImage(image=image)

#             self.video_label1.imgtk = imgtk
#             self.video_label1.configure(image=imgtk)

#     def update_rain_output(self):
#         while self.running:
#             ret, frame = self.cap.read()
#             if not ret: continue
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             image = Image.fromarray(frame_rgb).resize((256, 256), Image.Resampling.LANCZOS)
#             imgtk = ImageTk.PhotoImage(image=image)

#             self.video_label2.imgtk = imgtk
#             self.video_label2.configure(image=imgtk)

#     def update_processed_output(self):
#         while self.running:
#             ret, frame = self.cap.read()
#             if not ret: continue
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             processed_frame = self.apply_model(frame_rgb)
#             processed_image = Image.fromarray(processed_frame)
#             processed_imgtk = ImageTk.PhotoImage(image=processed_image)

#             self.video_label3.imgtk = processed_imgtk
#             self.video_label3.configure(image=processed_imgtk)

# if __name__ == "__main__":
#     root = ctk.CTk()
#     app = CameraApp(root)
#     root.mainloop()

import numpy as np
import cv2
import customtkinter as ctk
import torch
import threading
import os
from torchvision import transforms
from PIL import Image, ImageTk
# import chromadb
# from chromadb.config import Settings

# Load the pre-trained TorchScript model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = torch.jit.load("models/generator_model.pt", map_location=device)
generator.eval()

# # Initialize ChromaDB client and collection
# chroma_client = chromadb.Client(Settings(persist_directory="./chroma_storage", anonymized_telemetry=False))
# if "face_embeddings" not in chroma_client.list_collections():
#     collection = chroma_client.create_collection(name="face_embeddings")
# else:
#     collection = chroma_client.get_collection(name="face_embeddings")

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
        self.root = root
        self.root.title("Face Recognition in Rain")
        self.root.state("zoomed")

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.current_screen = None
        self.cap = cv2.VideoCapture(0)
        self.running = True

        # Navigation bar - moved to the left
        self.navbar = ctk.CTkFrame(self.root, width=200, fg_color="#1e1e2f", corner_radius=15)
        self.navbar.pack(side="left", fill="y", padx=10, pady=10)
        ctk.CTkButton(self.navbar, text="Live Feed", command=self.show_live_feed, width=180).pack(pady=10)
        ctk.CTkButton(self.navbar, text="Recognition", command=self.show_recognition, width=180).pack(pady=10)
        ctk.CTkButton(self.navbar, text="Register Face", command=self.show_register_face, width=180).pack(pady=10)
        ctk.CTkButton(self.navbar, text="Analytics", command=self.show_analytics, width=180).pack(pady=10)
        ctk.CTkButton(self.navbar, text="Settings", command=self.show_settings, width=180).pack(pady=10)

        # Container for the main content area
        self.container = ctk.CTkFrame(self.root)
        self.container.pack(side="right", expand=True, fill="both", padx=20, pady=20)

        self.show_live_feed()

    def clear_screen(self):
        for widget in self.container.winfo_children():
            widget.destroy()

    def show_live_feed(self):
        self.clear_screen()

        self.video_container = ctk.CTkFrame(self.container)
        self.video_container.pack(expand=True, pady=20)

        self.frame1 = self.create_video_frame("Live Feed")
        self.frame2 = self.create_video_frame("No Rain Output")
        self.frame3 = self.create_video_frame("Rain Augmented")

        self.video_label1 = self.create_video_label(self.frame1)
        self.video_label2 = self.create_video_label(self.frame2)
        self.video_label3 = self.create_video_label(self.frame3)

        self.frame1.grid(row=0, column=0, padx=15, pady=15)
        self.frame2.grid(row=0, column=1, padx=15, pady=15)
        self.frame3.grid(row=0, column=2, padx=15, pady=15)
        self.video_container.grid_columnconfigure((0, 1, 2), weight=1)

        self.running = True 
        self.live_thread = threading.Thread(target=self.update_live_feed)
        self.rain_thread = threading.Thread(target=self.update_rain_output)
        self.process_thread = threading.Thread(target=self.update_processed_output)

        self.live_thread.start()
        self.rain_thread.start()
        self.process_thread.start()

    def show_recognition(self):
        self.clear_screen()
        label = ctk.CTkLabel(self.container, text="Face Recognition & Confidence", font=("Arial", 20, "bold"))
        label.pack(pady=50)

    def show_register_face(self):
        self.clear_screen()

        self.register_frame = ctk.CTkFrame(self.container)
        self.register_frame.pack(pady=20)

        name_label = ctk.CTkLabel(self.register_frame, text="Full Name:", font=("Arial", 12))
        name_label.grid(row=0, column=0, padx=10, pady=10)
        self.name_entry = ctk.CTkEntry(self.register_frame, width=300)
        self.name_entry.grid(row=0, column=1, padx=10, pady=10)

        capture_button = ctk.CTkButton(self.register_frame, text="Register Face", command=self.capture_face_embedding, width=300)
        capture_button.grid(row=1, column=0, columnspan=2, pady=10)

        self.feedback_label = ctk.CTkLabel(self.container, text="", font=("Arial", 12))
        self.feedback_label.pack(pady=10)

        self.live_register_frame = ctk.CTkFrame(self.container)
        self.live_register_frame.pack()
        self.register_video_label = ctk.CTkLabel(self.live_register_frame, text="")
        self.register_video_label.pack()

        threading.Thread(target=self.update_register_feed).start()

    def show_analytics(self):
        self.clear_screen()
        label = ctk.CTkLabel(self.container, text="Recognition Analytics & Logs", font=("Arial", 20, "bold"))
        label.pack(pady=50)

    def show_settings(self):
        self.clear_screen()
        label = ctk.CTkLabel(self.container, text="Settings & Model Management", font=("Arial", 20, "bold"))
        label.pack(pady=50)

    def create_video_frame(self, text):
        frame = ctk.CTkFrame(self.video_container, width=256, height=256, fg_color="#2f2f3f", corner_radius=15)
        ctk.CTkLabel(frame, text=text, font=("SF Pro Display", 14, "bold")).pack(pady=5)
        return frame

    def create_video_label(self, frame):
        label = ctk.CTkLabel(frame, text="")
        label.pack()
        return label

    def apply_model(self, frame):
        img = Image.fromarray(frame)
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = generator(img)
            output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        processed_frame = (output * 255).astype("uint8")
        return processed_frame

    def update_register_feed(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame).resize((256, 256), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=image)
            if hasattr(self, 'register_video_label'):
                self.register_video_label.imgtk = imgtk
                self.register_video_label.configure(image=imgtk)
            self.current_frame = frame
            self.root.update()

    def capture_face_embedding(self):
        name = self.name_entry.get().strip()
        if not name:
            self.feedback_label.configure(text="Please enter a name", text_color="red")
            return
        if not hasattr(self, 'current_frame'):
            self.feedback_label.configure(text="No frame available", text_color="red")
            return

        embedding = get_face_embedding(self.current_frame)
        uid = f"{name}_{len(collection.get(ids=[]).get('ids', []))}"
        collection.add(embeddings=[embedding.tolist()], metadatas=[{"name": name}], ids=[uid])

        self.feedback_label.configure(text=f"Face embedding saved for {name}", text_color="green")

    def exit_app(self):
        self.running = False
        self.cap.release()
        self.root.quit()
        self.root.destroy()

    def update_live_feed(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret: continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb).resize((256, 256), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=image)

            self.video_label1.imgtk = imgtk
            self.video_label1.configure(image=imgtk)

    def update_rain_output(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret: continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb).resize((256, 256), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=image)

            self.video_label2.imgtk = imgtk
            self.video_label2.configure(image=imgtk)

    def update_processed_output(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret: continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = self.apply_model(frame_rgb)
            processed_image = Image.fromarray(processed_frame)
            processed_imgtk = ImageTk.PhotoImage(image=processed_image)

            self.video_label3.imgtk = processed_imgtk
            self.video_label3.configure(image=processed_imgtk)

if __name__ == "__main__":
    root = ctk.CTk()
    app = CameraApp(root)
    root.mainloop()

import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from ultralytics import YOLO

# --- Global Configuration ---
# Set the image size to match the training models
IMG_SIZE = 48
FONT = ("Arial", 12)
# Set the confidence threshold for drowsiness detection
DROWSINESS_THRESHOLD = 0.5
# Set confidence threshold for YOLO detections
YOLO_CONFIDENCE_THRESHOLD = 0.4

YOLO_CLASS_LABELS = ['Face', 'eyes']

class DrowsinessDetectorApp:
    def __init__(self, root):
        """Initializes the main application window and components."""
        self.root = root
        self.root.title("Drowsiness and Age Detection System (YOLOv8 Local)")
        self.root.geometry("800x600")
        self.root.configure(bg="#2c3e50")

        # --- Load Models and YOLO Model ---
        self.load_models()
        self.load_yolo_model()

        # --- GUI Elements ---
        # Create a frame for the video/image feed
        self.video_frame = tk.Frame(root, bg="#34495e", bd=5, relief="ridge")
        self.video_frame.pack(pady=20, padx=20, fill="both", expand=True)

        self.video_label = tk.Label(self.video_frame, bg="#34495e")
        self.video_label.pack(fill="both", expand=True)

        # Create a frame for the control buttons
        self.control_frame = tk.Frame(root, bg="#2c3e50")
        self.control_frame.pack(pady=10)

        # Buttons
        self.btn_start_video = tk.Button(
            self.control_frame,
            text="Start Webcam",
            command=self.start_video_stream,
            font=FONT,
            bg="#3498db",
            fg="white",
            relief="raised",
            bd=3,
            padx=10,
            pady=5,
        )
        self.btn_start_video.grid(row=0, column=0, padx=10)

        self.btn_stop_video = tk.Button(
            self.control_frame,
            text="Stop Webcam",
            command=self.stop_video_stream,
            font=FONT,
            bg="#e74c3c",
            fg="white",
            relief="raised",
            bd=3,
            padx=10,
            pady=5,
            state=tk.DISABLED,
        )
        self.btn_stop_video.grid(row=0, column=1, padx=10)

        self.btn_select_image = tk.Button(
            self.control_frame,
            text="Select Image",
            command=self.process_image,
            font=FONT,
            bg="#2ecc71",
            fg="white",
            relief="raised",
            bd=3,
            padx=10,
            pady=5,
        )
        self.btn_select_image.grid(row=0, column=2, padx=10)

        # Webcam
        self.cap = None
        self.running_video = False

        # --- Drowsiness detection state variables ---
        self.drowsiness_counter = 0
        self.drowsiness_threshold_frames = 15 
        self.drowsy_state_active = False

    def load_yolo_model(self):
        """Loads the locally-trained YOLOv8 model for face and eye detection."""
        try:
            model_path = os.path.join("src", "models", "best.pt")
            self.yolo_model = YOLO(model_path)
            # Fetch the class names from the model itself
            self.yolo_class_names = self.yolo_model.names
            
            # Verify that the expected class labels exist in the model
            for label in YOLO_CLASS_LABELS:
                if label not in self.yolo_class_names.values():
                    raise ValueError(f"Class '{label}' not found in the loaded YOLO model.")
                    
            print(f"YOLOv8 model loaded from {model_path} successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load YOLO model: {e}")
            self.root.destroy()
    
    def load_models(self):
        """Loads the custom-trained models."""
        try:
            drowsiness_model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Conv2D(128, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(1, activation='sigmoid')
            ])
            drowsiness_model.load_weights(os.path.join("src", "models", "drowsiness_model.weights.h5"))
            self.drowsiness_model = drowsiness_model
            
            age_model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Conv2D(128, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(256, activation='relu'),
                Dropout(0.5),
                Dense(1, activation='linear')
            ])
            age_model.load_weights(os.path.join("src", "models", "age_model.weights.h5"))
            self.age_model = age_model
            print("Custom models loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {e}")
            self.root.destroy()
    
    def process_frame(self, frame):
        """Processes a single frame for drowsiness and age detection using YOLO."""
        try:
            results = self.yolo_model(frame, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)
            
            faces = []
            eyes = []

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = self.yolo_class_names[cls]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    bbox = (int(x1), int(y1), int(x2-x1), int(y2-y1))
                    
                    if label.lower() == 'face':
                        faces.append(bbox)
                    elif label.lower() == 'eyes':
                        eyes.append(bbox)
        except Exception as e:
            # We don't show a messagebox here for every frame, as it would be too disruptive
            print(f"YOLO prediction error: {e}")
            return frame

        sleeping_people_count = 0
        ages_of_sleeping_people = []
        is_any_person_drowsy = False

        for (fx, fy, fw, fh) in faces:
            associated_eyes = []
            for (ex, ey, ew, eh) in eyes:
                if fx < ex and fy < ey and fx + fw > ex + ew and fy + fh > ey + eh:
                    associated_eyes.append((ex, ey, ew, eh))

            is_drowsy_face = False
            if len(associated_eyes) < 2:
                is_drowsy_face = True
            else:
                open_eye_count = 0
                for (ex, ey, ew, eh) in associated_eyes:
                    eye_roi = cv2.resize(cv2.cvtColor(frame[ey:ey+eh, ex:ex+ew], cv2.COLOR_BGR2GRAY), (IMG_SIZE, IMG_SIZE))
                    eye_roi = eye_roi.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0
                    prediction = self.drowsiness_model.predict(eye_roi, verbose=0)
                    if prediction[0][0] > DROWSINESS_THRESHOLD:
                        open_eye_count += 1
                
                if open_eye_count < 2:
                    is_drowsy_face = True
            
            if is_drowsy_face:
                is_any_person_drowsy = True
                sleeping_people_count += 1
                
                face_roi_age = cv2.resize(frame[fy:fy+fh, fx:fx+fw], (IMG_SIZE, IMG_SIZE))
                face_roi_age = face_roi_age.reshape(1, IMG_SIZE, IMG_SIZE, 3) / 255.0
                predicted_age = self.age_model.predict(face_roi_age, verbose=0)[0][0]
                ages_of_sleeping_people.append(int(predicted_age))

                cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 0, 255), 2)
                cv2.putText(frame, f"Drowsy! Age: {int(predicted_age)}", (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)
                cv2.putText(frame, "Awake", (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if is_any_person_drowsy:
            self.drowsiness_counter += 1
        else:
            self.drowsiness_counter = 0
            self.drowsy_state_active = False

        if self.drowsiness_counter >= self.drowsiness_threshold_frames and not self.drowsy_state_active:
            self.show_popup(sleeping_people_count, ages_of_sleeping_people)
            self.drowsy_state_active = True

        return frame

    def start_video_stream(self):
        """Starts the webcam video stream."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            return

        self.running_video = True
        self.btn_start_video.config(state=tk.DISABLED)
        self.btn_stop_video.config(state=tk.NORMAL)
        
        self.drowsiness_counter = 0
        self.drowsy_state_active = False

        self.update_video_feed()

    def update_video_feed(self):
        """Updates the video feed frame by frame."""
        if self.running_video:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                processed_frame = self.process_frame(frame)
                
                img = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)
                self.video_label.after(10, self.update_video_feed)
            else:
                self.stop_video_stream()
                messagebox.showerror("Error", "Failed to capture video.")
    
    def stop_video_stream(self):
        """Stops the webcam video stream."""
        if self.cap is not None:
            self.cap.release()
            self.running_video = False
            self.btn_start_video.config(state=tk.NORMAL)
            self.btn_stop_video.config(state=tk.DISABLED)
            self.video_label.config(image="")
            
            self.drowsiness_counter = 0
            self.drowsy_state_active = False
    
    def process_image(self):
        """Allows the user to select and process an image file."""
        self.stop_video_stream()
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            try:
                frame = cv2.imread(file_path)
                if frame is None:
                    raise FileNotFoundError("Could not read image file.")
                
                self.drowsiness_counter = 0
                self.drowsy_state_active = False
                
                processed_frame = self.process_frame(frame)

                img = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
                img_width, img_height = img.size
                aspect_ratio = img_width / img_height
                
                label_width = self.video_label.winfo_width()
                label_height = self.video_label.winfo_height()

                if aspect_ratio > (label_width / label_height):
                    new_width = label_width
                    new_height = int(new_width / aspect_ratio)
                else:
                    new_height = label_height
                    new_width = int(new_height * aspect_ratio)
                
                img = img.resize((new_width, new_height), Image.LANCZOS)
                
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process image: {e}")

    def show_popup(self, count, ages):
        """Displays a pop-up message with the count of sleeping people and their ages."""
        age_str = ", ".join(map(str, ages))
        message = f"{count} person(s) detected as sleeping!\nAges: {age_str}"
        messagebox.showinfo("Drowsiness Alert", message)

if __name__ == "__main__":
    root = tk.Tk()
    app = DrowsinessDetectorApp(root)
    root.mainloop()

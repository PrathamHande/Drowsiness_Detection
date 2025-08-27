# Import necessary libraries
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model

# --- Global Configuration ---
# Set the image size to match the training models
IMG_SIZE = 48
FONT = ("Arial", 12)
# Set the confidence threshold for drowsiness detection
DROWSINESS_THRESHOLD = 0.5


class DrowsinessDetectorApp:
    def __init__(self, root):
        """Initializes the main application window and components."""
        self.root = root
        self.root.title("Drowsiness and Age Detection System")
        self.root.geometry("800x600")
        self.root.configure(bg="#2c3e50")

        # --- Load Models and Classifiers ---
        self.load_models()
        self.load_cascades()

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

    def load_models(self):
        """Loads the pre-trained and custom-trained models."""
        try:
            # Rebuild the drowsiness model architecture to load the weights
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
            # path to load drowsiness model weights
            drowsiness_model.load_weights(os.path.join("src", "models", "drowsiness_model.weights.h5"))
            self.drowsiness_model = drowsiness_model
            
            # Rebuild the age model architecture
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
            # path to load age model weights
            age_model.load_weights(os.path.join("src", "models", "age_model.weights.h5"))
            self.age_model = age_model

            print("Models loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {e}")
            self.root.destroy()
    
    def load_cascades(self):
        """Loads the pre-trained Haar Cascade classifiers."""
        try:
            # path to the models directory
            cascades_path = os.path.join("src", "models")
            self.face_cascade = cv2.CascadeClassifier(os.path.join(cascades_path, "haarcascade_frontalface_default.xml"))
            self.eye_cascade = cv2.CascadeClassifier(os.path.join(cascades_path, "haarcascade_eye.xml"))
            if self.face_cascade.empty() or self.eye_cascade.empty():
                raise FileNotFoundError("Haar Cascade XML files not found or empty.")
            print("Haar Cascades loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load Haar Cascades: {e}")
            self.root.destroy()

    def _associate_eyes_with_faces(self, faces, eyes):
        """Associates eyes with the nearest face, returning a dictionary mapping face index to eyes and a list of unassociated eyes."""
        face_eye_map = {i: [] for i in range(len(faces))}
        unassociated_eyes = list(eyes)

        for i, (fx, fy, fw, fh) in enumerate(faces):
            face_center = (fx + fw // 2, fy + fh // 2)
            
            eyes_to_remove = []
            for j, (ex, ey, ew, eh) in enumerate(unassociated_eyes):
                eye_center = (ex + ew // 2, ey + eh // 2)

                # Check if the eye is within the face's bounding box with some margin
                if fx < eye_center[0] < fx + fw and fy < eye_center[1] < fy + fh:
                    face_eye_map[i].append((ex, ey, ew, eh))
                    eyes_to_remove.append(j)
            
            # Remove associated eyes from the unassociated list
            for j in sorted(eyes_to_remove, reverse=True):
                unassociated_eyes.pop(j)

        return face_eye_map, unassociated_eyes

    def process_frame(self, frame):
        """Processes a single frame for drowsiness and age detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces and eyes independently on the full frame
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        eyes = self.eye_cascade.detectMultiScale(gray, 1.3, 5)

        sleeping_people_count = 0
        ages_of_sleeping_people = []

        # Associate eyes with faces
        face_eye_map, unassociated_eyes = self._associate_eyes_with_faces(faces, eyes)
        
        # Corrected condition to check if both lists are empty
        if len(faces) == 0 and len(unassociated_eyes) == 0:
            self.drowsiness_counter = 0
            self.drowsy_state_active = False

        is_any_person_drowsy = False

        # Process detected faces
        for i, (x, y, w, h) in enumerate(faces):
            associated_eyes = face_eye_map[i]
            
            is_drowsy_face = False
            if len(associated_eyes) < 2:
                is_drowsy_face = True
            else:
                open_eye_count = 0
                for (ex, ey, ew, eh) in associated_eyes:
                    eye_roi = cv2.resize(gray[ey:ey+eh, ex:ex+ew], (IMG_SIZE, IMG_SIZE))
                    eye_roi = eye_roi.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0
                    prediction = self.drowsiness_model.predict(eye_roi, verbose=0)
                    if prediction[0][0] > DROWSINESS_THRESHOLD:
                        open_eye_count += 1
                
                if open_eye_count < 2:
                    is_drowsy_face = True

            if is_drowsy_face:
                is_any_person_drowsy = True
                sleeping_people_count += 1
                
                # Predict age for the drowsy person
                face_roi_age = cv2.resize(frame[y:y+h, x:x+w], (IMG_SIZE, IMG_SIZE))
                face_roi_age = face_roi_age.reshape(1, IMG_SIZE, IMG_SIZE, 3) / 255.0
                predicted_age = self.age_model.predict(face_roi_age, verbose=0)[0][0]
                ages_of_sleeping_people.append(int(predicted_age))

                # Draw a red rectangle and display the age
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, f"Drowsy! Age: {int(predicted_age)}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                # Draw a green rectangle for awake people
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Awake", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Process unassociated eyes (e.g., face is off-screen)
        if len(unassociated_eyes) >= 2:
            is_drowsy_unassociated = False
            open_eye_count = 0
            for (ex, ey, ew, eh) in unassociated_eyes:
                eye_roi = cv2.resize(gray[ey:ey+eh, ex:ex+ew], (IMG_SIZE, IMG_SIZE))
                eye_roi = eye_roi.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0
                prediction = self.drowsiness_model.predict(eye_roi, verbose=0)
                if prediction[0][0] > DROWSINESS_THRESHOLD:
                    open_eye_count += 1
            
            if open_eye_count < 2:
                is_drowsy_unassociated = True

            if is_drowsy_unassociated:
                is_any_person_drowsy = True
                sleeping_people_count += 1
                
                (ex, ey, ew, eh) = unassociated_eyes[0]
                cv2.rectangle(frame, (ex, ey), (unassociated_eyes[-1][0] + unassociated_eyes[-1][2], ey + eh), (0, 0, 255), 2)
                cv2.putText(frame, "Drowsy! (No Face)", (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
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
        
        # Reset counters when starting a new stream
        self.drowsiness_counter = 0
        self.drowsy_state_active = False

        self.update_video_feed()

    def update_video_feed(self):
        """Updates the video feed frame by frame."""
        if self.running_video:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1) # Mirror the image
                processed_frame = self.process_frame(frame)
                
                # Convert the frame to a format compatible with Tkinter
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
            
            # Reset counters when stopping the stream
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
                
                # Reset counters for image processing
                self.drowsiness_counter = 0
                self.drowsy_state_active = False
                
                processed_frame = self.process_frame(frame)

                # Resize the image to fit the display area
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
    # Create the main window instance
    root = tk.Tk()
    # Create an instance of the DrowsinessDetectorApp class
    app = DrowsinessDetectorApp(root)
    # Start the Tkinter event loop
    root.mainloop()

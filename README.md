# Drowsiness and Age Detection System

This project is a real-time computer vision application that detects drowsiness and predicts the age of individuals in a video stream or image file. It is built using Python and TensorFlow and features two distinct versions for object detection.

## Key Features

- **Real-time Drowsiness Detection:** The system monitors a person's eyes and classifies them as "Awake" or "Drowsy." It uses a counter to prevent false positives from normal blinking.
- **Multi-person Detection:** The application can detect and process multiple faces and their corresponding eyes in a single frame.
- **Age Prediction:** For each person detected as "Drowsy," the model predicts their age and displays it on the screen.
- **User Interface (GUI):** A clean graphical interface built with `tkinter` allows users to easily switch between live webcam feed and static image analysis.

## Project Versions

The project contains two versions of the main application, each using a different method for face and eye detection.

### 1. Haar Cascade Version (`main_app.py`)
This version uses traditional **Haar Cascade classifiers** for face and eye detection. It is a lightweight and classic approach that is easy to set up.

### 2. YOLOv8 Version (`new_main_app.py`)
This version uses a modern, deep learning-based **YOLOv8** model for detection. It offers significantly improved accuracy and robustness, especially with faces that are not perfectly aligned or are in challenging lighting conditions.

## Project Structure

Drowsiness_Detection/
├── data/
│   ├── raw/
│   │   ├── kaggle_drowsiness_dataset/
│   │   └── UTKFace_dataset/
├── src/
│   ├── models/
│   │   ├── haarcascade_frontalface_default.xml
│   │   ├── haarcascade_eye.xml
│   │   ├── best.pt
│   │   ├── drowsiness_model.weights.h5
│   │   └── age_model.weights.h5
│   ├── drowsiness_training.ipynb
│   ├── Yolo_model_training.ipynb
│   ├── main_app.py
│   └── new_main_app.py
└── requirements.txt

## Setup and Installation

### 1. Clone the Repository
Clone the project to your local machine

### 2. Create and activate a virtual environment

```bash
python -m venv venv
venv/Scripts/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

## How to Run the Applications

### 1. To Run the Haar Cascade Version:
```bash
python src/main_app.py
```
### 2. To Run the YOLOv8 Version
```bash
python src/new_main_app.py
```

## Download Datasets and Models

To get the project fully running, you will need to download the datasets and the custom-trained model weights.

Datasets: Download the Kaggle Drowsiness Detection and UTKFace datasets and place them in the data/raw/ directory.

Custom Models: The custom-trained model weights (.h5 files) are too large for GitHub. You can download them from the following link and place them in src/models/.

[Google Drive Link for Models: [INSERT_LINK_HERE](https://drive.google.com/drive/folders/1WlXQ-t1_JFRqe1yIYGnVN5iDu-kvyaB0?usp=drive_link)]
# 📷 Face Recognition Attendance Management System

A modern, desktop-based **Attendance Management System using Face Recognition** built with Python, OpenCV (LBPH Face Recognizer), CustomTkinter, and Pandas.

---

## ✨ Features

- ** Modern Dark GUI**: Sleek CustomTkinter interface with tabbed navigation, real-time live video viewports, and live status indicators.
- **👤 Student Registration**: Easy form to capture student details (ID, Name, Roll No, Department) and automatically collect 50 face samples via live webcam.
- **🧠 Model Training**: One-click training of OpenCV's **LBPH (Local Binary Patterns Histograms)** Face Recognizer on captured face datasets.
- **📸 Real-Time Face Recognition**: Continuous live tracking with green bounding boxes for recognized students and red bounding boxes for unknown faces.
- **⏱ Anti-Duplicate Log System**: Automatically marks attendance in real-time while preventing duplicate logs for students remaining in camera view.
- **📊 Attendance Dashboard**: Filter attendance logs by date, view real-time present student metric cards, and export daily attendance directly to **Excel (.xlsx)** and **CSV**.
- **📂 Student Directory**: Manage registered students and track dataset sample counts.

---

## 🛠️ Project Structure

```text
face recognisation system/
├── main.py                     # Main CustomTkinter Application Entry Point
├── modules/
│   ├── __init__.py
│   ├── attendance_logger.py    # CSV & Excel Attendance Database Handler
│   ├── dataset_creator.py      # Face Capture & Pre-processing (50 Samples)
│   ├── face_detector.py        # OpenCV Haar Cascade Face Detector
│   ├── face_recognizer.py      # Real-Time LBPH Face Prediction Engine
│   └── face_trainer.py         # Model Trainer (Generates trainer.yml)
├── StudentDetails/             # Contains student_details.csv
├── TrainingImage/              # Stored face samples (User.<ID>.<Name>.<Count>.jpg)
├── TrainingImageLabel/         # Stored trained model file (trainer.yml)
├── Attendance/                 # Daily Attendance CSV & Excel logs
└── README.md                   # Setup & User Guide
```

---

## 🚀 How to Run the Application

### 1. Prerequisites
Ensure Python 3.8+ is installed on your computer.

### 2. Install Dependencies
Open Terminal / Command Prompt in this folder and install required packages:
```bash
pip install opencv-contrib-python customtkinter pandas openpyxl pillow numpy
```

### 3. Launch the Application
Run `main.py`:
```bash
python main.py
```

---

## 📖 Step-by-Step Usage Guide

### Step 1: Register a New Student
1. Open the **"👤 Student Registration"** tab.
2. Enter the **Student ID** (e.g. `101`), **Full Name**, **Roll Number**, and select the **Department**.
3. Click **"1. Start Camera & Capture Dataset"**.
4. Look into the camera while subtle face movements (tilt head slightly) occur. The app will automatically capture 50 standardized face images.

### Step 2: Train the Model
1. Once 50 samples are captured (or from the **"📂 Student Directory"** tab), click **"2. Train Model"**.
2. The top-right badge will change to **"● Model Trained & Ready"**.

### Step 3: Take Real-Time Attendance
1. Switch to the **"📸 Mark Attendance"** tab.
2. Click **"▶ Start Attendance Tracker"**.
3. When registered students stand in front of the camera, the system will recognise them, display their name & match percentage, and mark attendance in today's log automatically.

### Step 4: View & Export Attendance Reports
1. Go to the **"📊 Attendance Dashboard"** tab.
2. Pick any date (default is today) to view all present students.
3. Click **"📥 Export Excel (.xlsx)"** to save attendance reports for your institution/school.

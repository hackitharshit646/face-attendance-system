import os
import io
import time
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Import backend modules
from modules.face_detector import FaceDetector
from modules.attendance_logger import AttendanceLogger
from modules.dataset_creator import DatasetCreator
from modules.face_trainer import FaceTrainer
from modules.face_recognizer import FaceRecognizer

# Optional WebRTC support for live video stream in browser
try:
    import av
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# -----------------------------------------------------------------------------
# Page Configuration & Custom CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Face Recognition Attendance System",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Dark theme tweaks */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        border: 1px solid #334155;
    }
    .status-badge-ready {
        background-color: #064e3b;
        color: #34d399;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
    }
    .status-badge-warning {
        background-color: #7f1d1d;
        color: #fca5a5;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Initialize Backend Instances
# -----------------------------------------------------------------------------
@st.cache_resource
def get_backend_modules():
    detector = FaceDetector()
    logger = AttendanceLogger()
    creator = DatasetCreator()
    trainer = FaceTrainer()
    recognizer = FaceRecognizer()
    return detector, logger, creator, trainer, recognizer

detector, logger, creator, trainer, recognizer = get_backend_modules()

# Always ensure recognizer model is reloaded if trained
recognizer.load_model()

# -----------------------------------------------------------------------------
# Sidebar Navigation & Status Header
# -----------------------------------------------------------------------------
st.sidebar.title("📷 Face Attendance System")
st.sidebar.markdown("---")

is_trained = trainer.is_trained()
if is_trained:
    st.sidebar.markdown('<span class="status-badge-ready">● Model Trained & Ready</span>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<span class="status-badge-warning">○ Model Not Trained</span>', unsafe_allow_html=True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
menu_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "📸 Mark Attendance",
        "👤 Student Registration",
        "🧠 Model Training",
        "📊 Attendance Dashboard",
        "📂 Student Directory"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption(f"📅 Current Date: {datetime.now().strftime('%Y-%m-%d')}")

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def process_single_frame(image_bytes, is_registration=False, student_id=None, student_name=None):
    """Processes a frame captured from camera input for detection/recognition."""
    # Convert bytes to cv2 numpy frame
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return None, []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detect_faces(gray)
    results = []

    for (x, y, w, h) in faces:
        if is_registration:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        else:
            # Prediction mode
            pred_id, dist, is_match = recognizer.predict(frame, (x, y, w, h))
            if is_match and pred_id is not None:
                student = logger.get_student_by_id(pred_id)
                name = student["Name"] if student else f"ID: {pred_id}"
                match_pct = int(max(0, min(100, 100 - dist)))
                label = f"{name} ({match_pct}%)"
                color = (0, 255, 0)
                
                # Mark attendance
                success, msg = logger.mark_attendance(pred_id, name)
                results.append({"id": pred_id, "name": name, "status": msg, "success": success})
            else:
                label = "Unknown Face"
                color = (0, 0, 255)
                results.append({"id": None, "name": "Unknown", "status": "Unknown face detected", "success": False})

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Convert BGR back to RGB for Streamlit display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb, results, faces

# WebRTC Video Processor Class
if WEBRTC_AVAILABLE:
    class AttendanceVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.detector = detector
            self.recognizer = recognizer
            self.logger = logger
            self.last_marked = {}

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detect_faces(gray)

            for (x, y, w, h) in faces:
                pred_id, dist, is_match = self.recognizer.predict(img, (x, y, w, h))
                if is_match and pred_id is not None:
                    student = self.logger.get_student_by_id(pred_id)
                    name = student["Name"] if student else f"ID: {pred_id}"
                    match_pct = int(max(0, min(100, 100 - dist)))
                    label = f"{name} ({match_pct}%)"
                    color = (0, 255, 0)

                    # Anti-duplicate cooldown (10s)
                    now = time.time()
                    if pred_id not in self.last_marked or (now - self.last_marked[pred_id]) > 10:
                        self.logger.mark_attendance(pred_id, name)
                        self.last_marked[pred_id] = now
                else:
                    label = "Unknown"
                    color = (0, 0, 255)

                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

# -----------------------------------------------------------------------------
# TAB 1: MARK ATTENDANCE
# -----------------------------------------------------------------------------
if menu_choice == "📸 Mark Attendance":
    st.title("📸 Real-Time Face Attendance Tracker")
    st.write("Position yourself in front of the webcam to mark attendance automatically.")

    if not is_trained:
        st.warning("⚠️ No trained face model found! Please register students and train the model first in the **🧠 Model Training** tab.")
    
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Webcam Feed")
        tracking_mode = st.radio("Select Camera Mode", ["📷 Photo / Snapshot Scan", "📹 Live WebRTC Stream"], horizontal=True)

        if tracking_mode == "📹 Live WebRTC Stream":
            if WEBRTC_AVAILABLE:
                rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
                webrtc_streamer(
                    key="attendance-streamer",
                    video_processor_factory=AttendanceVideoProcessor,
                    rtc_configuration=rtc_config,
                    media_stream_constraints={"video": True, "audio": False}
                )
            else:
                st.info("Live WebRTC requires `streamlit-webrtc`. Please use Snapshot mode or install dependencies.")
        else:
            img_file = st.camera_input("Scan Face via Camera")
            if img_file is not None:
                frame_rgb, results, faces = process_single_frame(img_file, is_registration=False)
                if frame_rgb is not None:
                    st.image(frame_rgb, caption="Processed Camera Frame", use_column_width=True)
                    for res in results:
                        if res["success"]:
                            st.success(f"✅ {res['status']}")
                        else:
                            st.warning(f"⚠️ {res['status']}")

    with col2:
        st.subheader("📋 Today's Attendance Log")
        today_records = logger.get_today_attendance()
        if today_records:
            df_today = pd.DataFrame(today_records)
            st.dataframe(df_today, use_container_width=True, height=400)
            st.metric("Total Present Today", len(df_today))
        else:
            st.info("No attendance logged yet today.")

# -----------------------------------------------------------------------------
# TAB 2: STUDENT REGISTRATION
# -----------------------------------------------------------------------------
elif menu_choice == "👤 Student Registration":
    st.title("👤 Register New Student")
    st.write("Fill in student details and capture face dataset photos.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Enter Details")
        reg_id = st.number_input("Student ID (Numeric)", min_value=1, step=1, value=101)
        reg_name = st.text_input("Full Name", placeholder="e.g. John Doe")
        reg_roll = st.text_input("Roll Number", placeholder="e.g. CS202601")
        reg_dept = st.selectbox("Department", ["Computer Science", "Information Technology", "Electronics", "Mechanical", "Civil", "Electrical"])

        if st.button("💾 Register / Save Student Profile", type="primary"):
            if not reg_name or not reg_roll:
                st.error("Please enter both Name and Roll Number!")
            else:
                logger.register_student(reg_id, reg_name, reg_roll, reg_dept)
                st.success(f"Student '{reg_name}' (ID: {reg_id}) registered successfully!")

        existing_samples = creator.get_image_count_for_student(reg_id)
        st.metric("Captured Face Samples", f"{existing_samples} / 50")

    with col2:
        st.subheader("2. Capture Face Samples")
        st.write("Take sample photos using your camera below to build the dataset for this student:")

        cam_img = st.camera_input("Capture Face Snapshot")
        if cam_img is not None:
            if not reg_name:
                st.error("Please enter Student Name before capturing face samples!")
            else:
                file_bytes = np.asarray(bytearray(cam_img.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector.detect_faces(gray)

                if len(faces) == 0:
                    st.warning("No face detected in the snapshot. Make sure your face is clearly visible!")
                else:
                    sample_num = creator.get_image_count_for_student(reg_id) + 1
                    saved_path = creator.save_face_sample(frame, faces[0], reg_id, reg_name, sample_num)
                    if saved_path:
                        st.success(f"Sample #{sample_num} saved successfully!")
                        st.rerun()

        st.markdown("---")
        st.write("Or Upload Face Images from File:")
        uploaded_files = st.file_uploader("Upload Face Images (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if uploaded_files:
            if not reg_name:
                st.error("Please enter Student Name first!")
            else:
                count = 0
                for up_file in uploaded_files:
                    up_bytes = np.asarray(bytearray(up_file.read()), dtype=np.uint8)
                    frame = cv2.imdecode(up_bytes, cv2.IMREAD_COLOR)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = detector.detect_faces(gray)
                    if len(faces) > 0:
                        s_num = creator.get_image_count_for_student(reg_id) + 1
                        if creator.save_face_sample(frame, faces[0], reg_id, reg_name, s_num):
                            count += 1
                st.success(f"Added {count} uploaded face sample(s) for Student ID {reg_id}!")
                st.rerun()

# -----------------------------------------------------------------------------
# TAB 3: MODEL TRAINING
# -----------------------------------------------------------------------------
elif menu_choice == "🧠 Model Training":
    st.title("🧠 Train Face Recognition Model")
    st.write("Train the LBPH Face Recognizer on all captured face samples in the dataset.")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    students_list = logger.get_all_students()
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TrainingImage")
    total_imgs = len([f for f in os.listdir(dataset_dir) if f.endswith('.jpg')]) if os.path.exists(dataset_dir) else 0

    with col1:
        st.metric("Registered Students", len(students_list))
    with col2:
        st.metric("Total Face Dataset Images", total_imgs)
    with col3:
        st.metric("Model Trained Status", "READY" if is_trained else "NOT TRAINED")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Start Model Training", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(current, total):
            pct = int((current / total) * 100)
            progress_bar.progress(pct)
            status_text.text(f"Processing image {current} / {total} ({pct}%)...")

        with st.spinner("Training model, please wait..."):
            success, n_images, n_students, message = trainer.train_classifier(progress_callback=update_progress)

        if success:
            st.success(f"🎉 {message}")
            recognizer.load_model()
            st.rerun()
        else:
            st.error(f"❌ Training Failed: {message}")

# -----------------------------------------------------------------------------
# TAB 4: ATTENDANCE DASHBOARD
# -----------------------------------------------------------------------------
elif menu_choice == "📊 Attendance Dashboard":
    st.title("📊 Attendance Dashboard & Reports")
    st.write("Filter attendance logs, view analytics, and export reports to Excel or CSV.")

    selected_date = st.date_input("Select Date for Attendance Log", datetime.now())
    date_str = selected_date.strftime("%Y-%m-%d")

    records = logger.get_attendance_by_date(date_str)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Present Students", len(records))
    with col2:
        registered_count = len(logger.get_all_students())
        attendance_pct = f"{round((len(records) / max(1, registered_count)) * 100, 1)}%" if registered_count else "N/A"
        st.metric("Attendance Rate", attendance_pct)

    st.markdown("<br>", unsafe_allow_html=True)

    if records:
        df = pd.DataFrame(records)
        st.subheader(f"Attendance Log for {date_str}")
        st.dataframe(df, use_container_width=True)

        # Export Buttons
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV Report",
                data=csv_data,
                file_name=f"Attendance_{date_str}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col_dl2:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Attendance')
            excel_data = excel_buffer.getvalue()

            st.download_button(
                label="📥 Download Excel (.xlsx) Report",
                data=excel_data,
                file_name=f"Attendance_{date_str}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    else:
        st.info(f"No attendance records found for {date_str}.")

# -----------------------------------------------------------------------------
# TAB 5: STUDENT DIRECTORY
# -----------------------------------------------------------------------------
elif menu_choice == "📂 Student Directory":
    st.title("📂 Student Directory & Dataset Overview")
    st.write("View all registered students and their dataset image count.")

    students = logger.get_all_students()
    if students:
        df_students = pd.DataFrame(students)
        # Calculate dataset image count for each student
        df_students["Face Samples"] = df_students["ID"].apply(lambda s_id: creator.get_image_count_for_student(s_id))
        st.dataframe(df_students, use_container_width=True)
    else:
        st.info("No registered students found. Register your first student in the **👤 Student Registration** tab.")

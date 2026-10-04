import os
import sys
import time
import threading
from datetime import datetime
import cv2
import PIL.Image, PIL.ImageTk
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Try importing CustomTkinter; fallback to standard tkinter if needed
try:
    import customtkinter as ctk
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    CTK_AVAILABLE = True
except ImportError:
    CTK_AVAILABLE = False

from modules.face_detector import FaceDetector
from modules.attendance_logger import AttendanceLogger
from modules.dataset_creator import DatasetCreator
from modules.face_trainer import FaceTrainer
from modules.face_recognizer import FaceRecognizer

class AttendanceSystemApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance Management System")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)

        # Initialize Backend Modules
        self.detector = FaceDetector()
        self.logger = AttendanceLogger()
        self.creator = DatasetCreator()
        self.trainer = FaceTrainer()
        self.recognizer = FaceRecognizer()

        # Camera & State Variables
        self.cap = None
        self.is_camera_running = False
        self.current_mode = None  # "registration" or "attendance"
        self.capture_count = 0
        self.max_samples = 50
        self.register_student_id = ""
        self.register_student_name = ""
        self.last_marked_time = {}  # anti-spam dictionary: student_id -> timestamp

        # UI Setup
        self._build_ui()
        self._start_clock()
        self.update_model_status_badge()

    def _build_ui(self):
        # Top Header Frame
        header = ctk.CTkFrame(self.root, height=70, corner_radius=0, fg_color="#1f2937")
        header.pack(side="top", fill="x")

        title_lbl = ctk.CTkLabel(
            header, 
            text="📷 Face Recognition Attendance Management System", 
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#f9fafb"
        )
        title_lbl.pack(side="left", padx=20, pady=15)

        # Right header panel (clock & model status badge)
        right_header = ctk.CTkFrame(header, fg_color="transparent")
        right_header.pack(side="right", padx=20, pady=10)

        self.lbl_clock = ctk.CTkLabel(
            right_header, 
            text="", 
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#38bdf8"
        )
        self.lbl_clock.pack(side="top", anchor="e")

        self.lbl_status_badge = ctk.CTkLabel(
            right_header, 
            text="Model Status: Checking...", 
            font=ctk.CTkFont(size=12),
            text_color="#9ca3af"
        )
        self.lbl_status_badge.pack(side="bottom", anchor="e")

        # Main Tabview Navigation
        self.tabview = ctk.CTkTabview(self.root, corner_radius=10)
        self.tabview.pack(fill="both", expand=True, padx=15, pady=15)

        self.tab_register = self.tabview.add(" 👤 Student Registration ")
        self.tab_attendance = self.tabview.add(" 📸 Mark Attendance ")
        self.tab_dashboard = self.tabview.add(" 📊 Attendance Dashboard ")
        self.tab_students = self.tabview.add(" 📂 Student Directory ")

        self._setup_registration_tab()
        self._setup_attendance_tab()
        self._setup_dashboard_tab()
        self._setup_directory_tab()

    def _start_clock(self):
        def update_time():
            now_str = datetime.now().strftime("%a, %b %d %Y | %I:%M:%S %p")
            self.lbl_clock.configure(text=now_str)
            self.root.after(1000, update_time)
        update_time()

    def update_model_status_badge(self):
        if self.trainer.is_trained():
            self.lbl_status_badge.configure(text="● Model Trained & Ready", text_color="#4ade80")
        else:
            self.lbl_status_badge.configure(text="○ Model Not Trained", text_color="#f87171")

    # =========================================================================
    # TAB 1: STUDENT REGISTRATION
    # =========================================================================
    def _setup_registration_tab(self):
        grid = ctk.CTkFrame(self.tab_register, fg_color="transparent")
        grid.pack(fill="both", expand=True, padx=10, pady=10)

        # Left Column: Inputs & Controls
        left_col = ctk.CTkFrame(grid, width=380)
        left_col.pack(side="left", fill="y", padx=10, pady=10)

        ctk.CTkLabel(left_col, text="Register New Student", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=15, pady=(15, 10))

        # Input fields
        ctk.CTkLabel(left_col, text="Student ID (Numeric):", font=ctk.CTkFont(size=13)).pack(anchor="w", padx=15, pady=(5, 0))
        self.entry_reg_id = ctk.CTkEntry(left_col, placeholder_text="e.g. 101")
        self.entry_reg_id.pack(fill="x", padx=15, pady=(0, 10))

        ctk.CTkLabel(left_col, text="Full Name:", font=ctk.CTkFont(size=13)).pack(anchor="w", padx=15, pady=(5, 0))
        self.entry_reg_name = ctk.CTkEntry(left_col, placeholder_text="e.g. John Doe")
        self.entry_reg_name.pack(fill="x", padx=15, pady=(0, 10))

        ctk.CTkLabel(left_col, text="Roll Number:", font=ctk.CTkFont(size=13)).pack(anchor="w", padx=15, pady=(5, 0))
        self.entry_reg_roll = ctk.CTkEntry(left_col, placeholder_text="e.g. CS202601")
        self.entry_reg_roll.pack(fill="x", padx=15, pady=(0, 10))

        ctk.CTkLabel(left_col, text="Department:", font=ctk.CTkFont(size=13)).pack(anchor="w", padx=15, pady=(5, 0))
        self.entry_reg_dept = ctk.CTkComboBox(left_col, values=["Computer Science", "Information Technology", "Electronics", "Mechanical", "Civil", "Electrical"])
        self.entry_reg_dept.pack(fill="x", padx=15, pady=(0, 15))

        # Progress bar for samples
        ctk.CTkLabel(left_col, text="Dataset Capture Progress:", font=ctk.CTkFont(size=13)).pack(anchor="w", padx=15, pady=(5, 0))
        self.progress_samples = ctk.CTkProgressBar(left_col)
        self.progress_samples.set(0)
        self.progress_samples.pack(fill="x", padx=15, pady=(5, 5))
        
        self.lbl_progress_count = ctk.CTkLabel(left_col, text="0 / 50 samples captured", font=ctk.CTkFont(size=12), text_color="#9ca3af")
        self.lbl_progress_count.pack(anchor="w", padx=15, pady=(0, 15))

        # Action Buttons
        btn_start_cap = ctk.CTkButton(
            left_col, text="1. Start Camera & Capture Dataset", 
            command=self.start_registration_capture,
            fg_color="#0284c7", hover_color="#0369a1"
        )
        btn_start_cap.pack(fill="x", padx=15, pady=6)

        btn_train = ctk.CTkButton(
            left_col, text="2. Train Model", 
            command=self.train_model_action,
            fg_color="#16a34a", hover_color="#15803d"
        )
        btn_train.pack(fill="x", padx=15, pady=6)

        btn_stop = ctk.CTkButton(
            left_col, text="Stop Camera", 
            command=self.stop_camera,
            fg_color="#dc2626", hover_color="#b91c1c"
        )
        btn_stop.pack(fill="x", padx=15, pady=6)

        # Right Column: Camera Viewport
        right_col = ctk.CTkFrame(grid)
        right_col.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(right_col, text="Live Camera Feed", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=15, pady=10)
        
        self.cam_canvas_reg = tk.Canvas(right_col, bg="#111827", highlightthickness=0)
        self.cam_canvas_reg.pack(fill="both", expand=True, padx=15, pady=15)

    def start_registration_capture(self):
        student_id = self.entry_reg_id.get().strip()
        name = self.entry_reg_name.get().strip()
        roll = self.entry_reg_roll.get().strip()
        dept = self.entry_reg_dept.get().strip()

        if not student_id or not student_id.isdigit():
            messagebox.showerror("Validation Error", "Please enter a valid numeric Student ID.")
            return
        if not name:
            messagebox.showerror("Validation Error", "Please enter Student Full Name.")
            return

        # Save/update student info in database
        self.logger.register_student(student_id, name, roll, dept)

        self.register_student_id = student_id
        self.register_student_name = name
        self.capture_count = 0
        self.progress_samples.set(0)
        self.lbl_progress_count.configure(text=f"0 / {self.max_samples} samples captured")

        self.start_camera(mode="registration")

    # =========================================================================
    # TAB 2: MARK ATTENDANCE
    # =========================================================================
    def _setup_attendance_tab(self):
        grid = ctk.CTkFrame(self.tab_attendance, fg_color="transparent")
        grid.pack(fill="both", expand=True, padx=10, pady=10)

        # Left Frame: Camera Feed (larger)
        left_col = ctk.CTkFrame(grid)
        left_col.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        header_frame = ctk.CTkFrame(left_col, fg_color="transparent")
        header_frame.pack(fill="x", padx=15, pady=10)

        ctk.CTkLabel(header_frame, text="Real-time Face Recognition Feed", font=ctk.CTkFont(size=16, weight="bold")).pack(side="left")

        btn_start_att = ctk.CTkButton(
            header_frame, text="▶ Start Attendance Tracker", 
            command=self.start_attendance_tracker,
            fg_color="#0284c7", hover_color="#0369a1", width=180
        )
        btn_start_att.pack(side="right", padx=5)

        btn_stop_att = ctk.CTkButton(
            header_frame, text="⏹ Stop Tracker", 
            command=self.stop_camera,
            fg_color="#dc2626", hover_color="#b91c1c", width=120
        )
        btn_stop_att.pack(side="right", padx=5)

        self.cam_canvas_att = tk.Canvas(left_col, bg="#111827", highlightthickness=0)
        self.cam_canvas_att.pack(fill="both", expand=True, padx=15, pady=15)

        # Right Frame: Today's Live Marked List
        right_col = ctk.CTkFrame(grid, width=350)
        right_col.pack(side="right", fill="y", padx=10, pady=10)

        ctk.CTkLabel(right_col, text="Today's Marked Attendance", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=15, pady=10)

        # Treeview table inside right_col
        tree_scroll = ttk.Scrollbar(right_col)
        tree_scroll.pack(side="right", fill="y", padx=(0, 10), pady=10)

        self.tree_live_att = ttk.Treeview(
            right_col, columns=("ID", "Name", "Time"), show="headings",
            yscrollcommand=tree_scroll.set, height=20
        )
        tree_scroll.config(command=self.tree_live_att.yview)

        self.tree_live_att.heading("ID", text="ID")
        self.tree_live_att.heading("Name", text="Name")
        self.tree_live_att.heading("Time", text="Time")

        self.tree_live_att.column("ID", width=60, anchor="center")
        self.tree_live_att.column("Name", width=140, anchor="w")
        self.tree_live_att.column("Time", width=90, anchor="center")

        self.tree_live_att.pack(fill="both", expand=True, padx=(10, 0), pady=10)
        self.refresh_live_attendance_table()

    def start_attendance_tracker(self):
        if not self.trainer.is_trained():
            messagebox.showwarning("Model Not Trained", "Please register student(s) and click 'Train Model' before marking attendance.")
            return

        self.recognizer.load_model()
        self.start_camera(mode="attendance")

    def refresh_live_attendance_table(self):
        for item in self.tree_live_att.get_children():
            self.tree_live_att.delete(item)

        records = self.logger.get_today_attendance()
        for r in reversed(records):
            self.tree_live_att.insert("", "end", values=(r.get("ID"), r.get("Name"), r.get("Time")))

    # =========================================================================
    # TAB 3: DASHBOARD & REPORTS
    # =========================================================================
    def _setup_dashboard_tab(self):
        main_frame = ctk.CTkFrame(self.tab_dashboard, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=15, pady=15)

        # Top Control Bar
        top_bar = ctk.CTkFrame(main_frame)
        top_bar.pack(fill="x", pady=(0, 15))

        ctk.CTkLabel(top_bar, text="Filter Date (YYYY-MM-DD):", font=ctk.CTkFont(size=13)).pack(side="left", padx=15, pady=10)
        self.entry_filter_date = ctk.CTkEntry(top_bar, width=140)
        self.entry_filter_date.insert(0, datetime.now().strftime("%Y-%m-%d"))
        self.entry_filter_date.pack(side="left", padx=5, pady=10)

        btn_load = ctk.CTkButton(top_bar, text="🔍 Load Records", command=self.load_dashboard_records, width=120)
        btn_load.pack(side="left", padx=10, pady=10)

        btn_export = ctk.CTkButton(top_bar, text="📥 Export Excel (.xlsx)", command=self.export_excel_action, fg_color="#16a34a", hover_color="#15803d", width=150)
        btn_export.pack(side="right", padx=15, pady=10)

        # Summary Metric Cards
        metrics_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        metrics_frame.pack(fill="x", pady=(0, 15))

        card1 = ctk.CTkFrame(metrics_frame, fg_color="#1f2937", corner_radius=10)
        card1.pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkLabel(card1, text="Registered Students", font=ctk.CTkFont(size=12), text_color="#9ca3af").pack(anchor="w", padx=15, pady=(10, 0))
        self.lbl_card_registered = ctk.CTkLabel(card1, text="0", font=ctk.CTkFont(size=24, weight="bold"), text_color="#38bdf8")
        self.lbl_card_registered.pack(anchor="w", padx=15, pady=(0, 10))

        card2 = ctk.CTkFrame(metrics_frame, fg_color="#1f2937", corner_radius=10)
        card2.pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkLabel(card2, text="Present Count", font=ctk.CTkFont(size=12), text_color="#9ca3af").pack(anchor="w", padx=15, pady=(10, 0))
        self.lbl_card_present = ctk.CTkLabel(card2, text="0", font=ctk.CTkFont(size=24, weight="bold"), text_color="#4ade80")
        self.lbl_card_present.pack(anchor="w", padx=15, pady=(0, 10))

        # Main Attendance Records Table
        table_frame = ctk.CTkFrame(main_frame)
        table_frame.pack(fill="both", expand=True)

        scroll = ttk.Scrollbar(table_frame)
        scroll.pack(side="right", fill="y", padx=(0, 5), pady=5)

        self.tree_dash = ttk.Treeview(
            table_frame, columns=("ID", "Name", "Date", "Time", "Status"), show="headings",
            yscrollcommand=scroll.set
        )
        scroll.config(command=self.tree_dash.yview)

        self.tree_dash.heading("ID", text="Student ID")
        self.tree_dash.heading("Name", text="Full Name")
        self.tree_dash.heading("Date", text="Date")
        self.tree_dash.heading("Time", text="Time")
        self.tree_dash.heading("Status", text="Status")

        self.tree_dash.column("ID", width=100, anchor="center")
        self.tree_dash.column("Name", width=250, anchor="w")
        self.tree_dash.column("Date", width=120, anchor="center")
        self.tree_dash.column("Time", width=120, anchor="center")
        self.tree_dash.column("Status", width=100, anchor="center")

        self.tree_dash.pack(fill="both", expand=True, padx=5, pady=5)

        self.load_dashboard_records()

    def load_dashboard_records(self):
        date_str = self.entry_filter_date.get().strip()
        for item in self.tree_dash.get_children():
            self.tree_dash.delete(item)

        records = self.logger.get_attendance_by_date(date_str)
        for r in records:
            self.tree_dash.insert("", "end", values=(r.get("ID"), r.get("Name"), r.get("Date"), r.get("Time"), r.get("Status")))

        # Update metric cards
        all_students = self.logger.get_all_students()
        self.lbl_card_registered.configure(text=str(len(all_students)))
        self.lbl_card_present.configure(text=str(len(records)))

    def export_excel_action(self):
        date_str = self.entry_filter_date.get().strip()
        success, msg = self.logger.export_to_excel(date_str)
        if success:
            messagebox.showinfo("Export Successful", msg)
        else:
            messagebox.showerror("Export Failed", msg)

    # =========================================================================
    # TAB 4: STUDENT DIRECTORY
    # =========================================================================
    def _setup_directory_tab(self):
        main_frame = ctk.CTkFrame(self.tab_students, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=15, pady=15)

        top_bar = ctk.CTkFrame(main_frame)
        top_bar.pack(fill="x", pady=(0, 15))

        ctk.CTkLabel(top_bar, text="Registered Students Directory", font=ctk.CTkFont(size=16, weight="bold")).pack(side="left", padx=15, pady=10)

        btn_refresh = ctk.CTkButton(top_bar, text="🔄 Refresh Directory", command=self.load_directory_records, width=140)
        btn_refresh.pack(side="right", padx=15, pady=10)

        btn_retrain = ctk.CTkButton(top_bar, text="🧠 Retrain Model", command=self.train_model_action, fg_color="#16a34a", hover_color="#15803d", width=140)
        btn_retrain.pack(side="right", padx=5, pady=10)

        table_frame = ctk.CTkFrame(main_frame)
        table_frame.pack(fill="both", expand=True)

        scroll = ttk.Scrollbar(table_frame)
        scroll.pack(side="right", fill="y", padx=(0, 5), pady=5)

        self.tree_dir = ttk.Treeview(
            table_frame, columns=("ID", "Name", "RollNo", "Department", "Images", "DateRegistered"), show="headings",
            yscrollcommand=scroll.set
        )
        scroll.config(command=self.tree_dir.yview)

        self.tree_dir.heading("ID", text="Student ID")
        self.tree_dir.heading("Name", text="Full Name")
        self.tree_dir.heading("RollNo", text="Roll Number")
        self.tree_dir.heading("Department", text="Department")
        self.tree_dir.heading("Images", text="Dataset Count")
        self.tree_dir.heading("DateRegistered", text="Date Registered")

        self.tree_dir.column("ID", width=90, anchor="center")
        self.tree_dir.column("Name", width=200, anchor="w")
        self.tree_dir.column("RollNo", width=120, anchor="center")
        self.tree_dir.column("Department", width=160, anchor="w")
        self.tree_dir.column("Images", width=110, anchor="center")
        self.tree_dir.column("DateRegistered", width=160, anchor="center")

        self.tree_dir.pack(fill="both", expand=True, padx=5, pady=5)
        self.load_directory_records()

    def load_directory_records(self):
        for item in self.tree_dir.get_children():
            self.tree_dir.delete(item)

        students = self.logger.get_all_students()
        for s in students:
            sid = s.get("ID")
            count = self.creator.get_image_count_for_student(sid)
            self.tree_dir.insert("", "end", values=(
                sid, s.get("Name"), s.get("RollNo"), s.get("Department"), f"{count} images", s.get("DateRegistered")
            ))

    # =========================================================================
    # CAMERA CONTROLLER & VIDEO PROCESSING
    # =========================================================================
    def start_camera(self, mode):
        if self.is_camera_running:
            self.stop_camera()

        self.current_mode = mode
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform.startswith('win') else cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not access webcam camera (Index 0). Please check your camera connection.")
            return

        self.is_camera_running = True
        self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self.video_thread.start()

    def stop_camera(self):
        self.is_camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def _video_loop(self):
        while self.is_camera_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Mirror frame horizontally for natural user interaction
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detect_faces(gray)

            if self.current_mode == "registration":
                self._process_registration_frame(frame, faces)
            elif self.current_mode == "attendance":
                self._process_attendance_frame(frame, faces)

            # Convert BGR OpenCV image to RGB PIL for Tkinter rendering
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = PIL.Image.fromarray(rgb_frame)

            # Resize to fit canvas
            if self.current_mode == "registration":
                canvas = self.cam_canvas_reg
            else:
                canvas = self.cam_canvas_att

            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            if canvas_width > 10 and canvas_height > 10:
                pil_img = pil_img.resize((canvas_width, canvas_height), PIL.Image.LANCZOS)

            photo_img = PIL.ImageTk.PhotoImage(image=pil_img)

            def update_canvas(p_img=photo_img, cvs=canvas):
                cvs.create_image(0, 0, image=p_img, anchor="nw")
                cvs.image = p_img

            self.root.after(0, update_canvas)
            time.sleep(0.03)  # ~30 FPS

    def _process_registration_frame(self, frame, faces):
        for (x, y, w, h) in faces:
            # Draw green bounding box around detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if self.capture_count < self.max_samples:
                saved = self.creator.save_face_sample(frame, (x, y, w, h), self.register_student_id, self.register_student_name, self.capture_count + 1)
                if saved:
                    self.capture_count += 1
                    pct = self.capture_count / self.max_samples
                    
                    def update_ui_progress(cnt=self.capture_count, p=pct):
                        self.progress_samples.set(p)
                        self.lbl_progress_count.configure(text=f"{cnt} / {self.max_samples} samples captured")
                    
                    self.root.after(0, update_ui_progress)

            # Text overlay on video
            status_txt = f"Capturing: {self.capture_count}/{self.max_samples}"
            cv2.putText(frame, status_txt, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if self.capture_count >= self.max_samples:
            self.is_camera_running = False
            def finish_capture():
                messagebox.showinfo("Dataset Complete", f"Successfully captured {self.max_samples} face samples for {self.register_student_name}!\n\nNext: Click 'Train Model' to activate face recognition.")
                self.load_directory_records()
            self.root.after(0, finish_finish_capture_wrapper, finish_capture)

    def _process_attendance_frame(self, frame, faces):
        for (x, y, w, h) in faces:
            student_id, dist, is_match = self.recognizer.predict(frame, (x, y, w, h))

            if is_match and student_id:
                student = self.logger.get_student_by_id(student_id)
                name = student.get("Name", "Student") if student else f"ID: {student_id}"
                label_text = f"{name} ({round(100 - dist)}%)"
                box_color = (0, 255, 0)  # Green

                # Anti-spam check: mark attendance max once per 10 seconds per student
                now_ts = time.time()
                last_ts = self.last_marked_time.get(student_id, 0)
                if now_ts - last_ts > 10:
                    success, msg = self.logger.mark_attendance(student_id, name)
                    self.last_marked_time[student_id] = now_ts
                    if success:
                        self.root.after(0, self.refresh_live_attendance_table)
                        self.root.after(0, self.load_dashboard_records)
            else:
                label_text = "Unknown Face"
                box_color = (0, 0, 255)  # Red

            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
            cv2.rectangle(frame, (x, y - 35), (x + w, y), box_color, cv2.FILLED)
            cv2.putText(frame, label_text, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def train_model_action(self):
        def run_training():
            success, total_imgs, total_students, msg = self.trainer.train_classifier()
            def on_complete():
                if success:
                    messagebox.showinfo("Training Successful", msg)
                    self.update_model_status_badge()
                    self.load_directory_records()
                else:
                    messagebox.showerror("Training Error", msg)
            self.root.after(0, on_complete)

        threading.Thread(target=run_training, daemon=True).start()

def finish_finish_capture_wrapper(fn):
    fn()

def main():
    if CTK_AVAILABLE:
        root = ctk.CTk()
    else:
        root = tk.Tk()
    
    app = AttendanceSystemApp(root)
    
    def on_closing():
        app.stop_camera()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()

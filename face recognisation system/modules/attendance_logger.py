import os
import csv
from datetime import datetime
import pandas as pd

class AttendanceLogger:
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.base_dir = base_dir
        self.student_dir = os.path.join(self.base_dir, "StudentDetails")
        self.attendance_dir = os.path.join(self.base_dir, "Attendance")
        
        os.makedirs(self.student_dir, exist_ok=True)
        os.makedirs(self.attendance_dir, exist_ok=True)
        
        self.student_file = os.path.join(self.student_dir, "student_details.csv")
        self._init_student_file()

    def _init_student_file(self):
        """Initializes the student_details.csv file if it doesn't exist."""
        if not os.path.exists(self.student_file):
            with open(self.student_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Name", "RollNo", "Department", "DateRegistered"])

    def get_today_csv_path(self):
        today_str = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.attendance_dir, f"Attendance_{today_str}.csv")

    def _init_today_attendance_file(self):
        file_path = self.get_today_csv_path()
        if not os.path.exists(file_path):
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Name", "Date", "Time", "Status"])
        return file_path

    def register_student(self, student_id, name, roll_no, department):
        """Registers or updates a student in student_details.csv."""
        students = self.get_all_students()
        date_reg = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Check if ID exists -> update, else append
        updated = False
        for s in students:
            if str(s["ID"]) == str(student_id):
                s["Name"] = name
                s["RollNo"] = roll_no
                s["Department"] = department
                s["DateRegistered"] = date_reg
                updated = True
                break
        
        if not updated:
            students.append({
                "ID": str(student_id),
                "Name": name,
                "RollNo": roll_no,
                "Department": department,
                "DateRegistered": date_reg
            })
            
        with open(self.student_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Name", "RollNo", "Department", "DateRegistered"])
            for s in students:
                writer.writerow([s["ID"], s["Name"], s["RollNo"], s["Department"], s["DateRegistered"]])
                
        return True

    def get_all_students(self):
        """Returns a list of dicts of all registered students."""
        students = []
        if os.path.exists(self.student_file):
            with open(self.student_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    students.append(row)
        return students

    def get_student_by_id(self, student_id):
        students = self.get_all_students()
        for s in students:
            if str(s["ID"]) == str(student_id):
                return s
        return None

    def mark_attendance(self, student_id, name=None):
        """
        Marks attendance for the student today if not already marked.
        Returns tuple: (success: bool, message: str)
        """
        if name is None:
            student = self.get_student_by_id(student_id)
            if student:
                name = student.get("Name", "Unknown")
            else:
                name = "Unknown"

        file_path = self._init_today_attendance_file()
        
        # Check if already marked today
        already_marked = False
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row["ID"]) == str(student_id):
                    already_marked = True
                    break

        if already_marked:
            return False, f"Attendance already recorded today for {name} (ID: {student_id})"

        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([str(student_id), name, date_str, time_str, "Present"])

        return True, f"Attendance marked for {name} at {time_str}"

    def get_today_attendance(self):
        """Returns today's attendance entries as a list of dicts."""
        file_path = self.get_today_csv_path()
        records = []
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(row)
        return records

    def get_attendance_by_date(self, date_str):
        file_path = os.path.join(self.attendance_dir, f"Attendance_{date_str}.csv")
        records = []
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(row)
        return records

    def export_to_excel(self, date_str=None):
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        csv_path = os.path.join(self.attendance_dir, f"Attendance_{date_str}.csv")
        xlsx_path = os.path.join(self.attendance_dir, f"Attendance_{date_str}.xlsx")
        
        if not os.path.exists(csv_path):
            return False, f"No attendance data found for {date_str}"
        
        df = pd.read_csv(csv_path)
        df.to_excel(xlsx_path, index=False)
        return True, f"Exported successfully to {os.path.basename(xlsx_path)}"

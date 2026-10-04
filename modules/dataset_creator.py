import os
import cv2

class DatasetCreator:
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_dir = base_dir
        self.dataset_dir = os.path.join(self.base_dir, "TrainingImage")
        os.makedirs(self.dataset_dir, exist_ok=True)

    def save_face_sample(self, frame, face_rect, student_id, name, sample_num):
        """
        Extracts face from frame using face_rect (x, y, w, h), pre-processes it,
        and saves to TrainingImage folder.
        Returns saved filename or None.
        """
        x, y, w, h = face_rect
        if w < 30 or h < 30:
            return None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_img = gray[y:y+h, x:x+w]
        
        # Pre-process: Equalize histogram & resize to standard dimensions
        face_img = cv2.equalizeHist(face_img)
        face_img = cv2.resize(face_img, (200, 200))
        
        # Format filename: User.<ID>.<Name>.<sample_num>.jpg
        clean_name = "".join(c for c in name if c.isalnum() or c in ('_', '-'))
        filename = f"User.{student_id}.{clean_name}.{sample_num}.jpg"
        filepath = os.path.join(self.dataset_dir, filename)
        
        cv2.imwrite(filepath, face_img)
        return filepath

    def get_image_count_for_student(self, student_id):
        """Counts how many dataset images exist for a student ID."""
        if not os.path.exists(self.dataset_dir):
            return 0
        count = 0
        prefix = f"User.{student_id}."
        for fname in os.listdir(self.dataset_dir):
            if fname.startswith(prefix) and fname.endswith(".jpg"):
                count += 1
        return count

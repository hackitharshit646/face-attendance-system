import os
import cv2
import numpy as np
from PIL import Image

class FaceTrainer:
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_dir = base_dir
        self.dataset_dir = os.path.join(self.base_dir, "TrainingImage")
        self.label_dir = os.path.join(self.base_dir, "TrainingImageLabel")
        os.makedirs(self.label_dir, exist_ok=True)
        self.trainer_file = os.path.join(self.label_dir, "trainer.yml")

    def train_classifier(self, progress_callback=None):
        """
        Trains LBPH Face Recognizer on all images in dataset_dir.
        Saves trained model to trainer.yml.
        Returns (success: bool, total_images: int, total_students: int, message: str)
        """
        if not hasattr(cv2, 'face') or not hasattr(cv2.face, 'LBPHFaceRecognizer_create'):
            return False, 0, 0, "OpenCV LBPH Face Recognizer module is missing. Please ensure opencv-contrib-python is installed."
        
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        if not os.path.exists(self.dataset_dir):
            return False, 0, 0, "TrainingImage directory does not exist."
            
        image_paths = [os.path.join(self.dataset_dir, f) for f in os.listdir(self.dataset_dir) if f.endswith('.jpg')]
        
        if not image_paths:
            return False, 0, 0, "No dataset images found in TrainingImage directory. Please capture student images first."
            
        faces = []
        ids = []
        
        total = len(image_paths)
        for idx, img_path in enumerate(image_paths):
            try:
                pil_img = Image.open(img_path).convert('L')
                img_np = np.array(pil_img, 'uint8')
                
                # Filename pattern: User.<ID>.<Name>.<sample_num>.jpg
                basename = os.path.basename(img_path)
                parts = basename.split('.')
                if len(parts) >= 4 and parts[0] == 'User':
                    student_id = int(parts[1])
                    faces.append(img_np)
                    ids.append(student_id)
            except Exception as e:
                print(f"Error reading image {img_path}: {e}")
                
            if progress_callback:
                progress_callback(idx + 1, total)

        if not faces or not ids:
            return False, 0, 0, "Could not extract valid face samples for training."
            
        recognizer.train(faces, np.array(ids))
        recognizer.write(self.trainer_file)
        
        unique_students = len(set(ids))
        return True, len(faces), unique_students, f"Successfully trained model with {len(faces)} images across {unique_students} student(s)."

    def is_trained(self):
        return os.path.exists(self.trainer_file)

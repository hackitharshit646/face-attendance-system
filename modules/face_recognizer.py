import os
import cv2

class FaceRecognizer:
    def __init__(self, base_dir=None, confidence_threshold=70.0):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_dir = base_dir
        self.trainer_file = os.path.join(self.base_dir, "TrainingImageLabel", "trainer.yml")
        self.confidence_threshold = confidence_threshold
        self.recognizer = None
        self.load_model()

    def load_model(self):
        """Loads trained LBPH model if available."""
        if hasattr(cv2, 'face') and hasattr(cv2.face, 'LBPHFaceRecognizer_create'):
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            if os.path.exists(self.trainer_file):
                try:
                    self.recognizer.read(self.trainer_file)
                    return True
                except Exception as e:
                    print(f"Error reading trainer.yml: {e}")
                    self.recognizer = None
        return False

    def is_model_loaded(self):
        return self.recognizer is not None and os.path.exists(self.trainer_file)

    def predict(self, frame, face_rect):
        """
        Predicts face identity from a frame and face bounding box (x, y, w, h).
        Returns tuple: (student_id: int/None, confidence_score: float, is_recognized: bool)
        Note: In LBPH, lower distance/confidence value means HIGHER similarity.
        Typically distance < 70 is a good match.
        """
        if not self.is_model_loaded():
            return None, 100.0, False
            
        x, y, w, h = face_rect
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_img = gray[y:y+h, x:x+w]
        
        # Preprocess same as dataset creation
        face_img = cv2.equalizeHist(face_img)
        face_img = cv2.resize(face_img, (200, 200))
        
        try:
            student_id, dist = self.recognizer.predict(face_img)
            # dist is LBPH distance (0 = exact match). Threshold e.g. 70
            is_match = dist < self.confidence_threshold
            return student_id, round(dist, 2), is_match
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 100.0, False

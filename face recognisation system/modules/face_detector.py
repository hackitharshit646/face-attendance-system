import os
import cv2

class FaceDetector:
    def __init__(self, cascade_path=None):
        """
        Initializes Haar Cascade Face Detector.
        Uses OpenCV default cascade path if custom path not provided.
        """
        local_xml = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "haarcascade_frontalface_default.xml")
        cv2_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        
        paths_to_try = []
        if cascade_path:
            paths_to_try.append(cascade_path)
        paths_to_try.extend([cv2_path, local_xml, "haarcascade_frontalface_default.xml"])
        
        self.face_cascade = cv2.CascadeClassifier()
        loaded = False
        for p in paths_to_try:
            if os.path.exists(p) and self.face_cascade.load(p):
                self.cascade_path = p
                loaded = True
                break

        if not loaded:
            print("Warning: Could not load Haar cascade XML file.")

    def detect_faces(self, gray_frame, scale_factor=1.2, min_neighbors=5, min_size=(60, 60)):
        """
        Detects faces in a grayscale frame.
        Returns a list of bounding boxes (x, y, w, h).
        """
        if self.face_cascade.empty():
            return []
        
        faces = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )
        return faces

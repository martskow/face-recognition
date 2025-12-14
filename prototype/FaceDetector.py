import cv2
import numpy as np
from PIL import Image

class FaceDetector:
    def __init__(self, dimension=160, method='haar'):
        self.dimension = dimension

        if method == 'haar':
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        else:
            raise NotImplementedError("Tylko metoda Haar jest zaimplementowana")

    def get_face(self, frame):
        """
        frame: RGB [0,1] lub BGR uint8
        zwraca wyciętą twarz w rozmiarze self.dimension lub None
        """
        # jeśli float [0,1], konwertujemy do uint8 BGR
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            # brak twarzy
            return None, "No face detected"
        elif len(faces) > 1:
            # więcej niż jedna twarz, możemy wybrać największą
            areas = [w * h for (x, y, w, h) in faces]
            idx = np.argmax(areas)
            x, y, w, h = faces[idx]
            warning = "Multiple faces detected, using largest"
        else:
            x, y, w, h = faces[0]
            warning = None

        face_img = frame[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, (self.dimension, self.dimension))
        face_img = Image.fromarray(face_img)
        # normalizacja [0,1]
        face_norm = np.array(face_img) / 255.0

        return face_norm, warning
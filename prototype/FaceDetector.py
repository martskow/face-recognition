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
        Zwraca:
        - face_norm: wycięta i znormalizowana twarz (do FaceNet)
        - coords: lista [x, y, w, h] (do Anti-Spoofing)
        - warning: komunikat o błędach/ostrzeżeniach
        """
        # Konwersja do uint8, jeśli obraz wejściowy to float [0, 1]
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)

        # Haar Cascade wymaga obrazu w skali szarości
        try:
            # Próba konwersji z RGB (jeśli tak przesyłasz z Flask/PIL)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            is_rgb = True
        except:
            # Jeśli obraz jest w BGR (standard OpenCV)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            is_rgb = False

        faces = self.face_cascade.detectMultiScale(
            frame_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            return None, None, "No face detected"

        # Wybór największej twarzy
        if len(faces) > 1:
            areas = [w * h for (x, y, w, h) in faces]
            idx = np.argmax(areas)
            x, y, w, h = faces[idx]
            warning = "Multiple faces detected, using largest"
        else:
            x, y, w, h = faces[0]
            warning = None

        # 1. Przygotowanie COORDS (dla Anti-Spoofing)
        coords = [int(x), int(y), int(w), int(h)]

        # 2. Przygotowanie FACE (dla FaceNet)
        face_img = frame[y:y + h, x:x + w]

        # Konwersja na RGB dla PIL (jeśli oryginał był w BGR)
        if not is_rgb:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        face_img = cv2.resize(face_img, (self.dimension, self.dimension))
        face_img_pil = Image.fromarray(face_img)

        # Normalizacja [0, 1]
        face_norm = np.array(face_img_pil) / 255.0

        return face_norm, coords, warning
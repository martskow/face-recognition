import cv2
import numpy as np

class ImageAcquisition:
    def __init__(self, camera_id=0, frame_size=(640, 480)):
        self.cap = cv2.VideoCapture(camera_id)
        self.frame_size = frame_size

        if not self.cap.isOpened():
            raise RuntimeError("Unable to access the camera")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.resize(frame, self.frame_size)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_norm = frame_rgb / 255.0

        return frame_norm

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__": # pragma: no cover
    acquisition = ImageAcquisition()

    while True:
        frame = acquisition.get_frame()
        if frame is None:
            break

        preview = (frame * 255).astype(np.uint8)
        preview = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)

        cv2.imshow("Akwizycja obrazu", preview)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    acquisition.release()
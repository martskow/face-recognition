import os
import cv2
from torch.utils.data import Dataset
from PIL import Image
import re


class FaceBinaryDataset(Dataset):
    def __init__(self, root_dir, dimension=160, transform=None):
        self.samples = []
        self.labels = []
        self.userid = []
        self.transform = transform
        self.dimension = dimension

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # root_dir = "dataset/"
        for label_str, label in [('authorized', 1), ('unauthorized', 0)]:
            dir_path = os.path.join(root_dir, label_str)
            for user_folder in os.listdir(dir_path):
                user_path = os.path.join(dir_path, user_folder)
                user = re.search(r'\d+', user_folder).group()
                for img_name in os.listdir(user_path):
                    img_path = os.path.join(user_path, img_name)
                    self.samples.append(img_path)
                    self.labels.append(label)
                    self.userid.append(user)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        userid = self.userid[idx]

        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            print(f"No face detected in: {img_path}")
            return None, None, None

        x, y, w, h = faces[0]
        face = img[y:y + h, x:x + w]
        face = cv2.resize(face, (self.dimension, self.dimension))

        face = Image.fromarray(face)

        if self.transform:
            face = self.transform(face)

        return face, label, userid

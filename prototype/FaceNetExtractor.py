from facenet_pytorch import InceptionResnetV1
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

class FaceNetExtractor:
    def __init__(self):
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

    def describe(self, face_img):
        """
        face_img: PIL.Image lub np.ndarray RGB [0,1]
        Zwraca wektor embedding
        """
        if isinstance(face_img, np.ndarray):
            # konwersja numpy -> PIL
            face_img = Image.fromarray((face_img*255).astype(np.uint8))

        input_tensor = self.preprocess(face_img).unsqueeze(0)
        with torch.no_grad():
            features = self.model(input_tensor)
        return features.squeeze().numpy()
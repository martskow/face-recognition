import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.models import Model
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image
import torch
import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from skimage import feature
from skimage import color
from skimage import feature
#bump
# 1. LBP
# https://pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
class LBPExtractor:
    def __init__(self, num_points=24, radius=8):
        self.num_points = num_points
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # Convert the original image to gray scale
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = feature.local_binary_pattern(img_gray, self.num_points,
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.num_points + 3),
                                 range=(0, self.num_points + 2))

        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        return hist

# 2. HOG
# https://www.geeksforgeeks.org/hog-feature-visualization-in-python-using-skimage/
class HOGExtractor:
    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def describe(self, image):
        # Convert the original image to gray scale
        img_gray = color.rgb2gray(image)
        # img_gray_resized = cv2.resize(img_gray, (256, 256))
        features, hog_image = feature.hog(img_gray, orientations=self.orientations,
                                          pixels_per_cell=self.pixels_per_cell, cells_per_block=self.cells_per_block,
                                          visualize=True)
        return features, hog_image


# 3. CNN MobileNetV3Small
# https://huggingface.co/qualcomm/MobileNet-v3-Small
class CNNExtractor:
    def __init__(self):
        base_model = MobileNetV3Small(weights='imagenet', include_top=False, pooling='avg')
        self.model = Model(inputs=base_model.input, outputs=base_model.output)

    def describe(self, image):
        image_resized = cv2.resize(image, (224, 224))
        image_array = np.expand_dims(image_resized, axis=0)
        image_preprocessed = preprocess_input(image_array)
        features = self.model.predict(image_preprocessed)
        return features.flatten()


# 4. FaceNet – InceptionResnetV1
# https://github.com/timesler/facenet-pytorch
class FaceNetExtractor:
    def __init__(self):
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        self.preprocess = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def describe(self, image):
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        input_tensor = self.preprocess(image_pil).unsqueeze(0)
        with torch.no_grad():
            features = self.model(input_tensor)
        return features.squeeze().numpy()


# ------------------------------
# Example usage
# ------------------------------
path = "C:/Users/marts/Downloads/ja.jpeg"
image = cv2.imread(path)

if __name__ == "__main__":
    path = "C:/Users/marts/Downloads/ja.jpeg"
    image = cv2.imread(path)

    # LBP
    lbp_extractor = LBPExtractor()
    lbp_hist = lbp_extractor.describe(image)
    plt.hist(lbp_hist)
    plt.title("LBP Histogram")
    plt.show()
    print("LBP histogram:", lbp_hist)

    # HOG
    hog_extractor = HOGExtractor()
    hog_features, hog_vis = hog_extractor.describe(image)
    
    print("HOG feature vector length:", len(hog_features))
    print("HOG feature vector:", hog_features)

    # CNN
    cnn_extractor = CNNExtractor()
    cnn_features = cnn_extractor.describe(image)
    print("CNN feature vector length:", cnn_features.shape)
    print("CNN feature vector:", cnn_features)

    # FaceNet
    facenet_extractor = FaceNetExtractor()
    facenet_features = facenet_extractor.describe(image)
    print("FaceNet feature vector length:", facenet_features.shape)
    print("FaceNet feature vector:", facenet_features)
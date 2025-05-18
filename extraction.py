import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.models import Model
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image
import torch
import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from skimage import color
from skimage import feature
from sklearn.base import BaseEstimator

# 1. LBP
# https://pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
class LBPExtractor(BaseEstimator):
    """
    Extracts Local Binary Pattern (LBP) features from an image.

    Attributes:
        num_points (int): Number of circularly symmetric neighbor set points.
        radius (int): Radius of circle.
    """
    def __init__(self, num_points=24, radius=8):
        self.num_points = num_points
        self.radius = radius

    def describe(self, image, eps=1e-7):
        """
        Computes the LBP histogram of the input image.

        Args:
            image (np.ndarray or PIL.Image): Input image (RGB or BGR).
            eps (float): Small value to prevent division by zero during normalization.

        Returns:
            np.ndarray: Normalized histogram of LBP features.
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert the original image to gray scale
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_gray = img_gray.astype(np.float32) / 255.0  # [0, 1]
        img_gray = (img_gray - 0.5) / 0.5  # [-1, 1]
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
class HOGExtractor(BaseEstimator):
    """
    Extracts Histogram of Oriented Gradients (HOG) features from an image.

    Attributes:
        orientations (int): Number of orientation bins.
        pixels_per_cell (tuple): Size (in pixels) of a cell.
        cells_per_block (tuple): Number of cells in each block.
    """
    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def describe(self, image):
        """
        Computes the HOG feature vector for the input image.

        Args:
            image (np.ndarray or PIL.Image): Input image (RGB or BGR).

        Returns:
            np.ndarray: Flattened HOG feature vector.
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Convert the original image to gray scale
        # img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #
        # img_gray = img_gray.astype(np.float32) / 255.0  # [0, 1]
        # img_gray = (img_gray - 0.5) / 0.5  # [-1, 1]

        # img_gray_resized = cv2.resize(img_gray, (256, 256))
        features = feature.hog(image, orientations=self.orientations,
                                          pixels_per_cell=self.pixels_per_cell, cells_per_block=self.cells_per_block,
                                          channel_axis=-1, feature_vector=True)
        return features


# 3. CNN MobileNetV3Small
# https://huggingface.co/qualcomm/MobileNet-v3-Small
class CNNExtractor:
    """
    Extracts high-level features from images using MobileNetV3Large pretrained on ImageNet.
    """
    def __init__(self):
        base_model = MobileNetV3Large(
            input_shape=None,
            alpha=1.0,
            minimalistic=False,
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            classes=1000,
            pooling='avg',
            dropout_rate=0.2,
            classifier_activation="softmax",
            include_preprocessing=True,
            name="MobileNetV3Large",
        )
        self.model = Model(inputs=base_model.input, outputs=base_model.output)

    def describe(self, image):
        """
        Extracts features from an image using the MobileNetV3Large CNN.

        Args:
            image (np.ndarray or PIL.Image): Input image (RGB).

        Returns:
            np.ndarray: Flattened feature vector from the CNN.
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        #image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32)
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32) / 255.0  # Skala do [0, 1]

        image = preprocess_input(image) # This function is also normalizing
        features = self.model.predict(image, verbose=0)
        return features.flatten()


# 4. FaceNet – InceptionResnetV1
# https://github.com/timesler/facenet-pytorch
class FaceNetExtractor:
    """
    Extracts facial features using the pretrained FaceNet (InceptionResnetV1) model.
    """
    def __init__(self):
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def describe(self, image):
        """
        Computes the FaceNet feature embedding for the input image.

        Args:
               image (PIL.Image or np.ndarray): Input face image.

        Returns:
               np.ndarray: Feature vector representing the face.
        """
        input_tensor = self.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            features = self.model(input_tensor)
        return features.squeeze().numpy()


if __name__ == "__main__":
    path = "C:/Users/macie/Downloads/face.jpg"
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
    hog_features = hog_extractor.describe(image)

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

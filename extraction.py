import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.models import Model
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image
import torch


path = "C:/Users/marts/Downloads/ja.jpeg"
image = cv2.imread(path)


# 3. CNN MobileNetV3Small
# https://huggingface.co/qualcomm/MobileNet-v3-Small
base_model = MobileNetV3Small(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

def extract_cnn_features(image):
    image_resized = cv2.resize(image, (224, 224))
    image_array = np.expand_dims(image_resized, axis=0)
    image_preprocessed = preprocess_input(image_array)
    features = model.predict(image_preprocessed)
    return features.flatten()

feat_cnn = extract_cnn_features(image)
print(feat_cnn)


# 4. FaceNet – InceptionResnetV1
# https://github.com/timesler/facenet-pytorch
model = InceptionResnetV1(pretrained='vggface2').eval()

preprocess = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]) #skaluje piksele do przedziału [-1, 1]
])

def extract_facenet_features(image):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    input_tensor = preprocess(image_pil).unsqueeze(0)  # dodanie wymiaru batcha
    with torch.no_grad():
        features = model(input_tensor)
    return features.squeeze().numpy()


feat_facenet = extract_facenet_features(image)
print(feat_facenet)



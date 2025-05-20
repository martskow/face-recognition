from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from extraction import LBPExtractor, HOGExtractor, CNNExtractor, FaceNetExtractor
from sklearn.base import BaseEstimator, TransformerMixin
from PIL import Image
import numpy as np


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, extractor):
        self.extractor = extractor

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for img in X:
            feat = self.extractor.describe(img)
            features.append(feat.flatten())
        return np.array(features)


def build_pipeline(extractor_name):
    extractor_map = {
        "lbp": lambda: LBPExtractor(),
        "hog": lambda: HOGExtractor(),
        "cnn": lambda: CNNExtractor(),
        "facenet": lambda: FaceNetExtractor(),
    }

    if extractor_name not in extractor_map:
        raise ValueError(f"Unknown extractor: {extractor_name}")

    extractor = extractor_map[extractor_name]()

    pipeline = Pipeline(
        steps=[
            ('features', FeatureExtractor(extractor)),
            ('standardScaler', StandardScaler()),
            ('classifier', SVC())
        ]
    )

    return pipeline

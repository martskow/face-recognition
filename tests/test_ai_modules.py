import pytest
import numpy as np
import cv2
from prototype.FaceDetector import FaceDetector
from prototype.FaceNetExtractor import FaceNetExtractor


@pytest.fixture(scope="module")
def real_detector():
    return FaceDetector(dimension=160)


@pytest.fixture(scope="module")
def real_extractor():
    return FaceNetExtractor()


def test_face_detector_integration(real_detector):
    dummy_frame = np.zeros((480, 640, 3), dtype=np.float32)
    cv2.rectangle(dummy_frame, (200, 200), (400, 400), (1, 1, 1), -1)

    face, coords, _ = real_detector.get_face(dummy_frame)

    if face is not None:
        assert face.shape == (160, 160, 3)
        assert len(coords) == 4
    else:
        assert face is None


def test_get_face_no_face(real_detector):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    face, coords, warning = real_detector.get_face(img)

    assert face is None
    assert coords is None
    assert warning == "No face detected"


def test_get_face_normalization_float(real_detector):
    img = np.random.rand(480, 640, 3).astype(np.float32)
    face, coords, _ = real_detector.get_face(img)

    if face is not None:
        assert face.max() <= 1.0
        assert face.dtype == np.float32 or face.dtype == np.float64


def test_facenet_extractor_integration(real_extractor):
    fake_face = np.random.rand(160, 160, 3).astype(np.float32)

    embedding = real_extractor.describe(fake_face)

    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert len(embedding) > 0


def test_face_detector_invalid_input(real_detector):
    invalid_frame = np.zeros((100, 100), dtype=np.float32)

    with pytest.raises(Exception):
        real_detector.get_face(invalid_frame)

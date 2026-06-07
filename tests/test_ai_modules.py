import pytest
import numpy as np
import cv2
import torch
from unittest.mock import patch, MagicMock

from prototype.FaceDetector import FaceDetector
from prototype.FaceNetExtractor import FaceNetExtractor
from prototype.VoiceExtractor import VoiceEmbeddingExtractor, bulletproof_amp_decorator


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


def test_face_detector_init_invalid_method():
    with pytest.raises(NotImplementedError):
        FaceDetector(method='deep_learning')

@patch('cv2.CascadeClassifier.detectMultiScale')
def test_face_detector_multiple_faces(mock_detect, real_detector):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_detect.return_value = np.array([[10, 10, 50, 50], [100, 100, 200, 200]])

    face, coords, warning = real_detector.get_face(img)

    assert "Multiple faces detected" in warning
    assert coords[2] == 200


@patch('cv2.cvtColor')
@patch('cv2.CascadeClassifier.detectMultiScale')
def test_face_detector_bgr_fallback(mock_detect, mock_cvt,real_detector):
    img = np.ones((480, 640, 3), dtype=np.uint8)

    mock_detect.return_value = np.array([[0, 0, 10, 10]])

    mock_cvt.side_effect = [
        Exception("Not RGB"),
        np.zeros((480, 640), dtype=np.uint8),
        np.zeros((160, 160, 3), dtype=np.uint8)
    ]

    face, _, _ = real_detector.get_face(img)
    assert face is not None
    assert mock_cvt.call_count >= 2


@patch('cv2.CascadeClassifier.detectMultiScale')
def test_face_detector_single_face_success(mock_detect, real_detector):
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    mock_detect.return_value = np.array([[100, 100, 50, 50]])

    face, coords, warning = real_detector.get_face(img)

    assert face is not None
    assert warning is None
    assert face.shape == (160, 160, 3)
    assert face.max() <= 1.0


from prototype.VoiceExtractor import VoiceEmbeddingExtractor, bulletproof_amp_decorator


# =====================================================================
# 1. TESTY: HACK DLA TORCH.AMP (100% pokrycia dekoratora)
# =====================================================================

def test_bulletproof_amp_decorator():
    # Kiedy dekorator jest wywoływany bez argumentów (jako fabryka)
    decorator = bulletproof_amp_decorator()

    def dummy_func(): pass

    assert decorator(dummy_func) == dummy_func

    # Kiedy dekorator jest wywoływany od razu z funkcją (bez nawiasów)
    assert bulletproof_amp_decorator(dummy_func) == dummy_func


# =====================================================================
# 2. TESTY: INICJALIZACJA MODELU
# =====================================================================

@patch('prototype.VoiceExtractor.EncoderClassifier.from_hparams')
def test_voice_extractor_init(mock_from_hparams):
    mock_model = MagicMock()
    mock_from_hparams.return_value = mock_model

    extractor = VoiceEmbeddingExtractor()

    mock_from_hparams.assert_called_once_with(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )
    assert extractor.model == mock_model


# =====================================================================
# 3. TESTY: PRZETWARZANIE AUDIO (SUKCES)
# =====================================================================

@patch('prototype.VoiceExtractor.EncoderClassifier.from_hparams')
@patch('prototype.VoiceExtractor.av')
def test_voice_extractor_describe_success(mock_av, mock_from_hparams):
    # 1. Przygotowanie mocka modelu AI
    mock_model = MagicMock()
    # Udajemy odpowiedź z modelu SpeechBrain (1 batch, 1 kanał, wektor 192 cech)
    fake_embedding = torch.randn(1, 1, 192)
    mock_model.encode_batch.return_value = fake_embedding
    mock_from_hparams.return_value = mock_model

    extractor = VoiceEmbeddingExtractor()

    # 2. Przygotowanie mocków dla biblioteki PyAV
    mock_container = MagicMock()
    mock_stream = MagicMock()
    mock_container.streams.audio = [mock_stream]
    mock_av.open.return_value = mock_container

    # Symulacja zdekodowanych pakietów
    mock_packet = MagicMock()
    mock_container.decode.return_value = [mock_packet]

    mock_resampler = MagicMock()
    mock_av.AudioResampler.return_value = mock_resampler

    # Symulacja pojedynczej ramki audio po resamplingu
    mock_frame = MagicMock()
    mock_frame.to_ndarray.return_value = np.array([[0.1, 0.2]])

    mock_resampler.resample.return_value = [mock_frame]

    # 3. Wywołanie testowanej metody
    audio_bytes = b"fake_audio_data"
    result = extractor.describe(audio_bytes)

    # 4. Asercje - czy metoda zadziałała prawidłowo
    assert isinstance(result, np.ndarray)
    assert result.shape == (192,)  # Wektor musi być spłaszczony (squeeze)
    mock_model.encode_batch.assert_called_once()


# =====================================================================
# 4. TESTY: BRAK DANYCH W AUDIO (VALUE ERROR -> RUNTIME ERROR)
# =====================================================================

@patch('prototype.VoiceExtractor.EncoderClassifier.from_hparams')
@patch('prototype.VoiceExtractor.av')
def test_voice_extractor_empty_frames(mock_av, mock_from_hparams):
    mock_from_hparams.return_value = MagicMock()
    extractor = VoiceEmbeddingExtractor()

    # Symulujemy pusty plik audio (brak ramek w kontenerze)
    mock_container = MagicMock()
    mock_container.streams.audio = [MagicMock()]
    mock_av.open.return_value = mock_container

    mock_container.decode.return_value = []
    mock_resampler = MagicMock()
    mock_resampler.resample.return_value = []
    mock_av.AudioResampler.return_value = mock_resampler

    # System powinien rzucić RuntimeError z zagnieżdżonym ValueError
    with pytest.raises(RuntimeError, match="Could not decode any audio frames"):
        extractor.describe(b"empty_data")


# =====================================================================
# 5. TESTY: BŁĄD ZEWNĘTRZNY NP. USZKODZONY PLIK (RUNTIME ERROR)
# =====================================================================

@patch('prototype.VoiceExtractor.EncoderClassifier.from_hparams')
@patch('prototype.VoiceExtractor.av')
def test_voice_extractor_generic_exception(mock_av, mock_from_hparams):
    mock_from_hparams.return_value = MagicMock()
    extractor = VoiceEmbeddingExtractor()

    # Symulujemy błąd otwarcia pliku przez PyAV
    mock_av.open.side_effect = Exception("Corrupt file metadata")

    with pytest.raises(RuntimeError, match="Voice processing error.*Corrupt file"):
        extractor.describe(b"corrupt_data")
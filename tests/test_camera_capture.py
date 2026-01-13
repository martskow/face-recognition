import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from prototype.camera_capture import ImageAcquisition


@pytest.fixture
def mock_video_capture():
    with patch('cv2.VideoCapture') as mock_cap:
        cap_instance = MagicMock()
        cap_instance.isOpened.return_value = True
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cap_instance.read.return_value = (True, dummy_frame)

        mock_cap.return_value = cap_instance
        yield cap_instance


def test_camera_init_success(mock_video_capture):
    acq = ImageAcquisition(camera_id=0)
    assert acq.cap.isOpened() is True


def test_camera_init_failure():
    with patch('cv2.VideoCapture') as mock_cap:
        cap_instance = MagicMock()
        cap_instance.isOpened.return_value = False
        mock_cap.return_value = cap_instance

        with pytest.raises(RuntimeError) as excinfo:
            ImageAcquisition(camera_id=0)
        assert "Unable to access the camera" in str(excinfo.value)


def test_get_frame_success(mock_video_capture):
    acq = ImageAcquisition()
    frame = acq.get_frame()

    assert frame is not None
    assert frame.shape == (480, 640, 3)
    assert frame.max() <= 1.0


def test_get_frame_fail(mock_video_capture):
    mock_video_capture.read.return_value = (False, None)
    acq = ImageAcquisition()
    assert acq.get_frame() is None


def test_camera_release(mock_video_capture):
    acq = ImageAcquisition()
    with patch('cv2.destroyAllWindows') as mock_destroy:
        acq.release()
        assert mock_video_capture.release.called
        assert mock_destroy.called
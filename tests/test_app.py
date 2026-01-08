from unittest.mock import patch
import numpy as np
import pytest

with patch('camera_capture.ImageAcquisition'), \
     patch('FaceDetector.FaceDetector'), \
     patch('FaceNetExtractor.FaceNetExtractor'), \
     patch('src.anti_spoof_predict.AntiSpoofPredict'):

    from prototype.camera_api import validate_user_data, is_face_distance_valid, app, User, encrypt_embedding, \
        decrypt_embedding, base64_to_image


@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    with app.app_context():
        with app.test_client() as client:
            yield client


def test_validate_user_data_correct():
    valid_data = {
        "first_name": "Name",
        "last_name": "Surname",
        "email": "Name@test.com",
        "password": "strongpassword123"
    }
    assert len(validate_user_data(valid_data)) == 0


def test_validate_user_data_invalid_name():
    invalid_data = {
        "first_name": "Name123",
        "last_name": "Surname",
        "email": "Name@test.com",
        "password": "strongpassword123"
    }
    errors = validate_user_data(invalid_data)
    assert len(errors) > 0
    assert "Name contains forbidden characters or is too short (2 characters minimum)" in errors[0]


def test_validate_user_data_invalid_surname():
    invalid_data = {
        "first_name": "Name",
        "last_name": "Surname!",
        "email": "Name@test.com",
        "password": "strongpassword123"
    }
    errors = validate_user_data(invalid_data)
    assert len(errors) > 0
    assert "Surname contains forbidden characters or is too short (2 characters minimum)" in errors[0]


def test_validate_user_data_invalid_email():
    invalid_data = {
        "first_name": "Name",
        "last_name": "Surname",
        "email": "Nametest.com",
        "password": "strongpassword123"
    }
    errors = validate_user_data(invalid_data)
    assert len(errors) > 0
    assert "Invalid email address syntax" in errors[0]


def test_validate_user_data_invalid_password():
    invalid_data = {
        "first_name": "Name",
        "last_name": "Surname",
        "email": "Name@test.com",
        "password": "weakpw"
    }
    errors = validate_user_data(invalid_data)
    assert len(errors) > 0
    assert "Password has to be at least 8 characters long" in errors[0]


def test_is_face_distance_valid_too_small():
    bbox = (0, 0, 100, 100)
    frame_shape = (1000, 500, 3)

    valid, ratio = is_face_distance_valid(bbox, frame_shape)

    assert valid is False
    assert ratio < 0.08


def test_is_face_distance_valid_perfect():
    bbox = (0, 0, 220, 220)
    frame_shape = (480, 640, 3)

    valid, ratio = is_face_distance_valid(bbox, frame_shape)

    assert valid is True
    assert 0.08 <= ratio <= 0.35


def test_is_face_distance_valid_too_big():
    bbox = (0, 0, 450, 450)
    frame_shape = (480, 640, 3)

    valid, ratio = is_face_distance_valid(bbox, frame_shape)

    assert valid is False
    assert ratio > 0.35


def test_register_validation_no_data(client):
    response = client.post('/register', json={})
    assert response.status_code in [400, 500]


def test_register_email_already_exists(client):
    with patch('prototype.camera_api.User.query') as mock_query:
        mock_query.return_value.filter_by.return_value.first.return_value = User(email="exists@test.pl")

        payload = {
            "first_name": "Jan",
            "last_name": "Kowalski",
            "email": "exists@test.pl",
            "password": "password123"
        }
        response = client.post('/register', json=payload)

        assert response.status_code == 400
        assert "E-mail already exists" in response.get_json()['message']

def test_login_user_not_found(client):
    with patch('prototype.camera_api.User.query') as mock_query:
        mock_query.filter_by.return_value.first.return_value = None

        response = client.post('/login', json={
            "email": "nieistnieje@test.pl",
            "password": "password123"
        })

        assert response.status_code == 400
        assert response.get_json()['message'] == "Wrong email or password"


def test_login_wrong_password(client):
    fake_user = User(
        email="jan@test.pl",
        password="hashed_password_here"
    )

    with patch('prototype.camera_api.User.query') as mock_query, \
            patch('prototype.camera_api.check_password_hash', return_value=False):
        mock_query.filter_by.return_value.first.return_value = fake_user

        response = client.post('/login', json={
            "email": "jan@test.pl",
            "password": "zle_haslo"
        })

        assert response.status_code == 400
        assert "Wrong email" in response.get_json()['message']


@patch('prototype.camera_api.camera.get_frame')
@patch('prototype.camera_api.detector.get_face')
@patch('prototype.camera_api.facenet.describe')
@patch('prototype.camera_api.decrypt_embedding')
@patch('prototype.camera_api.parse_model_name')
@patch('prototype.camera_api.image_cropper.crop')
@patch('prototype.camera_api.anti_spoof_engine.predict')
def test_login_biometric_mismatch(mock_predict, mock_crop, mock_parse, mock_decrypt, mock_describe, mock_get_face,
                                  mock_get_frame, client):
    fake_user = User(id=1, email="test@test.pl", password="hash")

    mock_get_frame.return_value = np.zeros((480, 640, 3))
    mock_get_face.return_value = (np.zeros((160, 160, 3)), (100, 100, 200, 200), None)

    mock_parse.return_value = (80, 80, "name", 1.0)
    mock_crop.return_value = np.zeros((80, 80, 3))
    mock_predict.return_value = np.array([[0.0, 1.0, 0.0]])

    mock_describe.return_value = np.array([0.1, 0.2])
    mock_decrypt.return_value = np.array([0.9, 0.9])

    with patch('prototype.camera_api.User.query') as mock_query, \
            patch('prototype.camera_api.check_password_hash', return_value=True), \
            patch('prototype.camera_api.is_face_distance_valid', return_value=(True, 0.15)), \
            patch('os.listdir', return_value=['model.pth']):
        mock_query.return_value.filter_by.return_value.first.return_value = fake_user

        response = client.post('/login', json={"email": "test@test.pl", "password": "haslo"})

        assert response.status_code == 401
        assert "Face does not match" in response.get_json()['message']


@patch('prototype.camera_api.camera.get_frame')
@patch('prototype.camera_api.detector.get_face')
@patch('prototype.camera_api.parse_model_name')
@patch('prototype.camera_api.image_cropper.crop')
@patch('prototype.camera_api.anti_spoof_engine.predict')
def test_login_spoofing_detected(mock_predict, mock_crop, mock_parse, mock_get_face, mock_get_frame, client):
    fake_user = User(id=1, email="test@test.pl", password="hash")

    mock_get_frame.return_value = np.zeros((480, 640, 3))
    mock_get_face.return_value = (np.zeros((160, 160, 3)), (100, 100, 200, 200), None)

    mock_parse.return_value = (80, 80, "name", 1.0)
    mock_crop.return_value = np.zeros((80, 80, 3))
    mock_predict.return_value = np.array([[1.0, 0.0, 0.0]])

    with patch('prototype.camera_api.User.query') as mock_query, \
            patch('prototype.camera_api.check_password_hash', return_value=True), \
            patch('prototype.camera_api.is_face_distance_valid', return_value=(True, 0.15)), \
            patch('os.listdir', return_value=['model.pth']):
        mock_query.return_value.filter_by.return_value.first.return_value = fake_user

        response = client.post('/login', json={"email": "test@test.pl", "password": "haslo"})

        assert response.status_code == 403
        assert "Spoofing detected" in response.get_json()['message']


def test_encryption_decryption_consistency():
    original_embedding = np.random.rand(128).tolist()

    encrypted = encrypt_embedding(original_embedding)
    decrypted = decrypt_embedding(encrypted)

    np.testing.assert_array_almost_equal(original_embedding, decrypted)


def test_base64_to_image_malformed_string():
    invalid_data = "not-a-base64-string"
    with pytest.raises(Exception):
        base64_to_image(invalid_data)


@patch('prototype.camera_api.camera.get_frame')
@patch('prototype.camera_api.detector.get_face')
@patch('prototype.camera_api.is_face_distance_valid')
@patch('prototype.camera_api.anti_spoof_engine.predict')
def test_login_anti_spoofing_crash(mock_predict, mock_dist, mock_get_face, mock_get_frame, client):
    fake_user = User(id=1, email="test@test.pl", password="hash")
    with patch('prototype.camera_api.User.query') as mock_query, \
            patch('prototype.camera_api.check_password_hash', return_value=True):
        mock_query.return_value.filter_by.return_value.first.return_value = fake_user

        mock_get_frame.return_value = np.zeros((480, 640, 3))
        mock_get_face.return_value = (np.zeros((160, 160, 3)), (100, 100, 200, 200), None)
        mock_dist.return_value = (True, 0.20)

        mock_predict.side_effect = Exception("Plik modelu .pth jest uszkodzony")

        with patch('os.listdir', return_value=['model.pth']):
            response = client.post('/login', json={"email": "test@test.pl", "password": "haslo"})

            assert response.status_code == 500
            assert "Anti-spoofing error" in response.get_json()['message']


@patch('prototype.camera_api.camera.get_frame')
def test_login_camera_error(mock_get_frame, client):
    mock_get_frame.return_value = None

    fake_user = User(id=1, email="test@test.pl", password="hashed_password")
    with patch('prototype.camera_api.User.query') as mock_query, \
            patch('prototype.camera_api.check_password_hash', return_value=True):

        mock_query.return_value.filter_by.return_value.first.return_value = fake_user

        response = client.post('/login', json={"email": "test@test.pl", "password": "haslo"})
        assert response.status_code == 500
        assert "Camera access error" in response.get_json()['message']

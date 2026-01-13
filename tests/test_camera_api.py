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


@patch('prototype.camera_api.base64_to_image')
@patch('prototype.camera_api.detector.get_face')
@patch('prototype.camera_api.facenet.describe')
@patch('prototype.camera_api.db.session')
def test_register_success(mock_db, mock_describe, mock_get_face, mock_base64, client):
    mock_base64.return_value = np.zeros((160, 160, 3))
    mock_get_face.return_value = (np.zeros((160, 160, 3)), (0, 0, 10, 10), None)
    mock_describe.return_value = np.random.rand(128)

    with patch('prototype.camera_api.User.query') as mock_query:
        mock_query.filter_by.return_value.first.return_value = None

        payload = {
            "first_name": "Name",
            "last_name": "Surname",
            "email": "name@surname.pl",
            "password": "StrongPassword123!",
            "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        }

        response = client.post('/register', json=payload)

        assert response.status_code == 200
        assert "Registration successful!" in response.get_json()['message']
        assert mock_db.add.called
        assert mock_db.commit.called


@patch('prototype.camera_api.base64_to_image')
@patch('prototype.camera_api.detector.get_face')
def test_register_face_not_detected(mock_get_face, mock_base64, client):
    mock_base64.return_value = np.zeros((160, 160, 3))
    mock_get_face.return_value = (None, None, "No face detected")

    with patch('prototype.camera_api.User.query') as mock_query:
        mock_query.filter_by.return_value.first.return_value = None
        payload = {"first_name": "Name",
                   "last_name": "Surname",
                   "email": "name@surname.pl",
                   "password": "StrongPassword123!",
                   "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="}
        response = client.post('/register', json=payload)

        assert response.status_code == 400
        assert "Face not detected" in response.get_json()['message']


@patch('prototype.camera_api.camera.get_frame')
@patch('prototype.camera_api.detector.get_face')
@patch('prototype.camera_api.is_face_distance_valid')
def test_login_face_distance_errors(mock_dist, mock_get_face, mock_get_frame, client):
    fake_user = User(email="dist@test.pl", password="hash")
    mock_get_frame.return_value = np.zeros((480, 640, 3))
    mock_get_face.return_value = (np.zeros((160, 160, 3)), (0, 0, 10, 10), None)

    with patch('prototype.camera_api.User.query') as mock_query, \
            patch('prototype.camera_api.check_password_hash', return_value=True):
        mock_query.filter_by.return_value.first.return_value = fake_user

        mock_dist.return_value = (False, 0.99)
        res_close = client.post('/login', json={"email": "dist@test.pl", "password": "p"})
        assert "too close" in res_close.get_json()['message']

        mock_dist.return_value = (False, 0.01)
        res_far = client.post('/login', json={"email": "dist@test.pl", "password": "p"})
        assert "too far" in res_far.get_json()['message']

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
@patch('prototype.camera_api.is_face_distance_valid')
def test_login_face_distance_too_close(mock_dist, mock_get_face, mock_get_frame, client):
    fake_user = User(email="name@surname.pl", password="StrongPassword123!")
    mock_get_frame.return_value = np.zeros((480, 640, 3))
    mock_get_face.return_value = (np.zeros((160, 160, 3)), (0, 0, 10, 10), None)

    mock_dist.return_value = (False, 0.8)

    with patch('prototype.camera_api.User.query') as mock_query, \
            patch('prototype.camera_api.check_password_hash', return_value=True):
        mock_query.filter_by.return_value.first.return_value = fake_user

        response = client.post('/login', json={"email": "name@surname.pl", "password": "StrongPassword123!"})
        assert response.status_code == 400
        assert "Face too close" in response.get_json()['message']


@patch('prototype.camera_api.camera.get_frame')
@patch('prototype.camera_api.detector.get_face')
@patch('prototype.camera_api.is_face_distance_valid')
def test_login_face_distance_too_far(mock_dist, mock_get_face, mock_get_frame, client):
    fake_user = User(email="name@surname.pl", password="StrongPassword123!")
    mock_get_frame.return_value = np.zeros((480, 640, 3))
    mock_get_face.return_value = (np.zeros((160, 160, 3)), (0, 0, 10, 10), None)

    mock_dist.return_value = (False, 0.01)

    with patch('prototype.camera_api.User.query') as mock_query, \
            patch('prototype.camera_api.check_password_hash', return_value=True):
        mock_query.filter_by.return_value.first.return_value = fake_user

        response = client.post('/login', json={"email": "name@surname.pl", "password": "StrongPassword123!"})
        assert response.status_code == 400
        assert "Face too far" in response.get_json()['message']


@patch('prototype.camera_api.camera.get_frame')
@patch('prototype.camera_api.detector.get_face')
def test_login_face_not_detected(mock_get_face, mock_get_frame, client):
    mock_get_frame.return_value = np.zeros((480, 640, 3))
    mock_get_face.return_value = (None, None, "No face detected")

    fake_user = User(email="name@surname.pl", password="StrongPassword123!")

    with patch('prototype.camera_api.User.query') as mock_query, \
            patch('prototype.camera_api.check_password_hash', return_value=True):
        mock_query.filter_by.return_value.first.return_value = fake_user

        response = client.post('/login', json={"email": "name@surname.pl", "password": "StrongPassword123!"})

        assert response.status_code == 400
        assert "Face not detected" in response.get_json()['message']


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
        mock_query.filter_by.return_value.first.return_value = fake_user

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


def test_base64_to_image_success():
    valid_base64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    img = base64_to_image(valid_base64)

    assert isinstance(img, np.ndarray)
    assert img.shape[2] == 3
    assert img.max() <= 1.0


@patch('prototype.camera_api.camera.get_frame')
@patch('prototype.camera_api.detector.get_face')
def test_video_feed_stream(mock_get_face, mock_get_frame, client):
    mock_get_frame.return_value = np.zeros((480, 640, 3))
    mock_get_face.return_value = (None, None, "No face detected")

    response = client.get('/video')
    assert response.status_code == 200

    chunk = next(response.response)
    assert b'--frame' in chunk
    assert b'Content-Type: image/jpeg' in chunk


@patch('prototype.camera_api.camera.get_frame')
@patch('prototype.camera_api.detector.get_face')
def test_generate_frames_full_logic(mock_get_face, mock_get_frame):
    from prototype.camera_api import generate_frames

    mock_get_frame.side_effect = [None, np.zeros((480, 640, 3)), np.zeros((480, 640, 3))]
    mock_get_face.side_effect = [(None, None, "No face"), (np.zeros((160, 160, 3)), [0, 0, 10, 10], None)]

    gen = generate_frames()

    chunk1 = next(gen)
    chunk2 = next(gen)

    assert b'--frame' in chunk1
    assert b'--frame' in chunk2

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


@patch('prototype.camera_api.camera.get_frame')
@patch('prototype.camera_api.detector.get_face')
@patch('prototype.camera_api.is_face_distance_valid')
@patch('prototype.camera_api.parse_model_name')
@patch('prototype.camera_api.image_cropper.crop')
@patch('prototype.camera_api.anti_spoof_engine.predict')
@patch('prototype.camera_api.facenet.describe')
@patch('prototype.camera_api.decrypt_embedding')
@patch('prototype.camera_api.db.session')
def test_login_success_path(mock_db, mock_decrypt, mock_describe, mock_predict,
                            mock_crop, mock_parse, mock_dist, mock_get_face,
                            mock_get_frame, client):

    fake_user = User(id=1, first_name="Name", last_name="Surname", email="name@surname.pl",
                     password="StrongPassword123!")

    mock_get_frame.return_value = np.zeros((480, 640, 3))
    mock_get_face.return_value = (np.zeros((160, 160, 3)), (0, 0, 10, 10), None)
    mock_parse.return_value = (80, 80, "model_name", 1.0)
    mock_crop.return_value = np.zeros((80, 80, 3))
    mock_dist.return_value = (True, 0.2)
    mock_predict.return_value = np.array([[0.0, 1.0, 0.0]])
    mock_describe.return_value = np.array([0.1, 0.2])
    mock_decrypt.return_value = np.array([0.1, 0.2])

    with patch('prototype.camera_api.User.query') as mock_query, \
            patch('prototype.camera_api.check_password_hash', return_value=True), \
            patch('os.listdir', return_value=['model_1.pth']):

        mock_query.filter_by.return_value.first.return_value = fake_user

        response = client.post('/login', json={"email": "name@surname.pl", "password": "StrongPassword123!"})

        assert response.status_code == 200
        assert "Login successful" in response.get_json()['message']

def test_simple_views(client):
    assert client.get('/').status_code == 200
    assert client.get('/register_view').status_code == 200


def test_dashboard_redirect_if_no_session(client):
    response = client.get('/dashboard')
    assert response.status_code == 302
    assert '/' in response.location


def test_endpoints_coverage(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 1
        sess['first_name'] = "Name"
        sess['last_name'] = "Surname"
        sess['email'] = "name@surname.pl"

    response_dash = client.get('/dashboard')
    assert response_dash.status_code == 200

    response_status = client.get('/face_status')
    assert response_status.status_code == 200
    assert "face_detected" in response_status.get_json()

    response_logout = client.get('/logout')
    assert response_logout.status_code == 302

def test_logout(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 1

    response = client.get('/logout')
    assert response.status_code == 302
    with client.session_transaction() as sess:
        assert 'user_id' not in sess
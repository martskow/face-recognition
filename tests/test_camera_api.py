from unittest.mock import patch
import numpy as np
import pytest
from prototype.camera_api import LoginHistory
import datetime
import base64

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


@patch('prototype.camera_api.base64_to_audio')
@patch('prototype.camera_api.voice_extractor.describe')
@patch('prototype.camera_api.base64_to_image')
@patch('prototype.camera_api.detector.get_face')
@patch('prototype.camera_api.facenet.describe')
@patch('prototype.camera_api.db.session')
def test_register_success(mock_db, mock_describe, mock_get_face, mock_base64, mock_voice_describe, mock_base64_audio,
                          client):
    mock_base64.return_value = np.zeros((160, 160, 3))
    mock_get_face.return_value = (np.zeros((160, 160, 3)), (0, 0, 10, 10), None)
    mock_describe.return_value = np.random.rand(128)

    mock_base64_audio.return_value = b'fake_audio_bytes'
    mock_voice_describe.return_value = np.random.rand(128)

    with patch('prototype.camera_api.User.query') as mock_query:
        mock_query.filter_by.return_value.first.return_value = None

        payload = {
            "first_name": "Name",
            "last_name": "Surname",
            "email": "name@surname.pl",
            "password": "StrongPassword123!",
            "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
            "audio": "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="
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


@patch('prototype.camera_api.check_security_threats')
@patch('prototype.camera_api.camera.get_frame')
@patch('prototype.camera_api.detector.get_face')
@patch('prototype.camera_api.is_face_distance_valid')
def test_login_face_distance_errors(mock_dist, mock_get_face, mock_get_frame, mock_security, client):
    fake_user = User(id=98, email="dist@test.pl", password="hash", is_active=True, require_voice_auth=False)
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
    fake_user = User(id=1, email="name@surname.pl", password="StrongPassword123!", is_active=True, require_voice_auth=False)
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
    fake_user = User(id=1, email="name@surname.pl", password="StrongPassword123!", is_active=True, require_voice_auth=False)
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

    fake_user = User(id=1, email="name@surname.pl", password="StrongPassword123!", is_active=True, require_voice_auth=False)

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
    fake_user = User(id=1, email="test@test.pl", password="hash", is_active=True, require_voice_auth=False)

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
                     password="StrongPassword123!", is_active=True, require_voice_auth=False)

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


@patch('prototype.camera_api.LoginHistory.query')
def test_brute_force_lockdown(mock_history_query, client):
    fake_user = User(id=1, email="brute@test.pl", password="hash", is_active=True)

    fake_logs = [
        LoginHistory(user_id=1, status="Failed (400: Wrong credentials)")
        for _ in range(5)
    ]

    mock_history_query.filter.return_value.all.return_value = fake_logs

    with patch('prototype.camera_api.User.query') as mock_user_query, \
            patch('prototype.camera_api.db.session'):
        mock_user_query.filter_by.return_value.first.return_value = fake_user

        with patch('prototype.camera_api.check_password_hash', return_value=False):
            response = client.post('/login', json={
                "email": "brute@test.pl",
                "password": "wrong_password"
            })

            assert response.status_code == 400

            assert fake_user.is_active is False


@patch('prototype.camera_api.LoginHistory.query')
def test_spoofing_lockdown(mock_history_query, client):
    fake_user = User(id=2, email="spoof@test.pl", password="hash", is_active=True)

    fake_logs = [
        LoginHistory(user_id=2, status="Failed 403: Spoofing detected")
        for _ in range(3)
    ]

    mock_history_query.filter.return_value.all.return_value = fake_logs

    with patch('prototype.camera_api.User.query') as mock_user_query, \
            patch('prototype.camera_api.db.session'):
        mock_user_query.filter_by.return_value.first.return_value = fake_user

        with patch('prototype.camera_api.check_password_hash', return_value=False):
            client.post('/login', json={"email": "spoof@test.pl", "password": "wrong"})

            assert fake_user.is_active is False


def test_admin_export_report_success(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 99

    admin_user = User(id=99, email="admin@test.pl", is_admin=True)

    fake_logs = [
        LoginHistory(id=1, user_id=2, timestamp=datetime.datetime(2026, 6, 6, 12, 0, 0), ip_address="192.168.0.1",
                     user_agent="PyTest", status="Passed"),
        LoginHistory(id=2, user_id=3, timestamp=datetime.datetime(2026, 6, 6, 12, 5, 0), ip_address="192.168.0.2",
                     user_agent="PyTest", status="Failed (Spoofing)")
    ]

    with patch('prototype.camera_api.User.query') as mock_user_query, \
            patch('prototype.camera_api.LoginHistory.query') as mock_history_query:
        mock_user_query.get.return_value = admin_user

        mock_history_query.order_by.return_value.all.return_value = fake_logs

        response = client.get('/api/admin/export-report')

        assert response.status_code == 200
        assert "text/csv" in response.headers["Content-Type"]
        assert "attachment; filename=" in response.headers["Content-Disposition"]

        csv_content = response.data.decode('utf-8')
        assert "ID_Zdarzenia;ID_Uzytkownika;" in csv_content
        assert "192.168.0.1" in csv_content
        assert "Failed (Spoofing)" in csv_content


def test_base64_to_audio_conversion():
    from prototype.camera_api import base64_to_audio
    dummy_bytes = b"dummy_audio"
    b64_str = base64.b64encode(dummy_bytes).decode('utf-8')

    # Z prefiksem
    assert base64_to_audio(f"data:audio/wav;base64,{b64_str}") == dummy_bytes
    # Bez prefiksu
    assert base64_to_audio(b64_str) == dummy_bytes


def test_base64_to_cv2_img_success():
    from prototype.camera_api import base64_to_cv2_img
    valid_b64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    img = base64_to_cv2_img(valid_b64)
    assert img is not None
    assert img.shape == (1, 1, 3)


def test_base64_to_cv2_img_invalid():
    from prototype.camera_api import base64_to_cv2_img
    assert base64_to_cv2_img("niepoprawny_ciąg_znaków_b64") is None


@patch('prototype.camera_api.check_password_hash')
def test_login_check_success(mock_check_hash, client):
    fake_user = User(id=1, email="test@test.pl", password="hash", is_active=True, require_voice_auth=True)
    with patch('prototype.camera_api.User.query') as mock_query:
        mock_query.filter_by.return_value.first.return_value = fake_user
        mock_check_hash.return_value = True

        res = client.post('/api/login-check', json={"email": "test@test.pl", "password": "pass"})
        assert res.status_code == 200
        assert res.get_json()['valid'] is True
        assert res.get_json()['require_voice'] is True


@patch('prototype.camera_api.check_password_hash')
def test_login_check_blocked_account(mock_check_hash, client):
    fake_user = User(id=1, email="blocked@test.pl", password="hash", is_active=False)
    with patch('prototype.camera_api.User.query') as mock_query:
        mock_query.filter_by.return_value.first.return_value = fake_user
        mock_check_hash.return_value = True

        res = client.post('/api/login-check', json={"email": "blocked@test.pl", "password": "pass"})
        assert res.status_code == 403
        assert "zablokowane" in res.get_json()['message']


def test_update_profile_success(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 1

    fake_user = User(id=1, first_name="Old", last_name="Old")
    with patch('prototype.camera_api.User.query') as mock_query, \
            patch('prototype.camera_api.db.session'):
        mock_query.get.return_value = fake_user

        res = client.put('/api/user/profile', json={
            "first_name": "New",
            "last_name": "Newer",
            "password": "StrongPassword123!"
        })
        assert res.status_code == 200
        assert fake_user.first_name == "New"
        assert fake_user.last_name == "Newer"


def test_update_profile_no_auth(client):
    res = client.put('/api/user/profile', json={})
    assert res.status_code == 41


def test_get_user_history(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 1

    fake_log = LoginHistory(timestamp=datetime.datetime.now(), ip_address="127.0.0.1", status="Passed")
    with patch('prototype.camera_api.LoginHistory.query') as mock_query:
        mock_query.filter_by.return_value.order_by.return_value.all.return_value = [fake_log]

        res = client.get('/api/user/history')
        assert res.status_code == 200
        assert len(res.get_json()) == 1
        assert res.get_json()[0]['status'] == "Passed"


def test_toggle_voice_auth(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 1

    fake_user = User(id=1, require_voice_auth=True)
    with patch('prototype.camera_api.User.query') as mock_query, \
            patch('prototype.camera_api.db.session'):
        mock_query.get.return_value = fake_user

        res = client.post('/api/user/toggle-voice')
        assert res.status_code == 200
        assert fake_user.require_voice_auth is False
        assert res.get_json()['require_voice'] is False


@patch('prototype.camera_api.base64_to_cv2_img')
@patch('prototype.camera_api.detector.get_face')
@patch('prototype.camera_api.facenet.describe')
@patch('prototype.camera_api.base64_to_audio')
@patch('prototype.camera_api.voice_extractor.describe')
def test_reinit_biometrics_success(mock_voice_desc, mock_b64_audio, mock_face_desc, mock_get_face, mock_b64_img,
                                   client):
    with client.session_transaction() as sess:
        sess['user_id'] = 1

    fake_user = User(id=1)
    with patch('prototype.camera_api.User.query') as mock_query, \
            patch('prototype.camera_api.db.session'):
        mock_query.get.return_value = fake_user

        mock_b64_img.return_value = np.zeros((160, 160, 3), dtype=np.uint8)
        mock_get_face.return_value = (np.zeros((160, 160, 3)), (0, 0, 10, 10), None)
        mock_face_desc.return_value = np.array([0.1, 0.2])

        mock_b64_audio.return_value = b'audio_bytes'
        mock_voice_desc.return_value = np.array([0.3, 0.4])

        res = client.post('/api/user/reinit_biometrics', json={
            "image": "dummy_img_base64",
            "audio": "dummy_audio_base64"
        })

        assert res.status_code == 200
        assert "Twarz" in res.get_json()['message']
        assert "Glos" in res.get_json()['message']


def test_admin_panel_view(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 99

    fake_admin = User(id=99, is_admin=True)
    with patch('prototype.camera_api.User.query') as mock_query, \
            patch('prototype.camera_api.SystemConfig.query') as mock_conf, \
            patch('prototype.camera_api.db.session'), \
            patch('prototype.camera_api.render_template') as mock_render:
        mock_query.get.return_value = fake_admin
        mock_conf.first.return_value = None
        mock_render.return_value = "Admin HTML"

        res = client.get('/admin/panel')
        assert res.status_code == 200
        assert mock_render.called


def test_admin_get_users(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 99

    fake_admin = User(id=99, is_admin=True, email="admin@test.pl", first_name="A", last_name="A")
    fake_user = User(id=1, is_admin=False, email="user@test.pl", first_name="U", last_name="U")

    with patch('prototype.camera_api.User.query') as mock_query:
        mock_query.get.return_value = fake_admin
        mock_query.all.return_value = [fake_admin, fake_user]

        res = client.get('/api/admin/users')
        assert res.status_code == 200
        data = res.get_json()
        assert len(data) == 2
        assert data[1]['email'] == "user@test.pl"


def test_admin_toggle_user_status(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 99

    fake_admin = User(id=99, is_admin=True)
    fake_user = User(id=1, email="test@test.pl", is_active=True)

    with patch('prototype.camera_api.User.query') as mock_query, \
            patch('prototype.camera_api.db.session'):
        mock_query.get.return_value = fake_admin
        mock_query.get_or_404.return_value = fake_user

        res = client.post('/api/admin/users/1/toggle')
        assert res.status_code == 200
        assert fake_user.is_active is False

        mock_query.get_or_404.return_value = fake_admin
        res_self = client.post('/api/admin/users/99/toggle')
        assert res_self.status_code == 400


def test_admin_delete_user(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 99

    fake_admin = User(id=99, is_admin=True)
    fake_user = User(id=1, email="del@test.pl")

    with patch('prototype.camera_api.User.query') as mock_query, \
            patch('prototype.camera_api.LoginHistory.query'), \
            patch('prototype.camera_api.db.session') as mock_db:
        mock_query.get.return_value = fake_admin
        mock_query.get_or_404.return_value = fake_user

        res = client.delete('/api/admin/users/1')
        assert res.status_code == 200
        assert mock_db.delete.called

        mock_query.get_or_404.return_value = fake_admin
        res_self = client.delete('/api/admin/users/99')
        assert res_self.status_code == 400


def test_admin_update_config(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 99

    fake_admin = User(id=99, is_admin=True)
    from prototype.camera_api import SystemConfig
    fake_config = SystemConfig(face_threshold=0.6, voice_threshold=0.45)

    with patch('prototype.camera_api.User.query') as mock_query, \
            patch('prototype.camera_api.SystemConfig.query') as mock_conf, \
            patch('prototype.camera_api.db.session'):
        mock_query.get.return_value = fake_admin
        mock_conf.first.return_value = fake_config

        res = client.post('/api/admin/config', json={"face_threshold": 0.8, "voice_threshold": 0.5})
        assert res.status_code == 200
        assert fake_config.face_threshold == 0.8
        assert fake_config.voice_threshold == 0.5


def test_admin_security_report(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 99

    fake_admin = User(id=99, is_admin=True)

    with patch('prototype.camera_api.User.query') as mock_query, \
            patch('prototype.camera_api.LoginHistory.query') as mock_hist, \
            patch('prototype.camera_api.db.session.query') as mock_db_query:
        mock_query.get.return_value = fake_admin

        mock_hist.filter.return_value.count.return_value = 12

        fake_block = LoginHistory(user_id=2, timestamp=datetime.datetime.now(),
                                  status="BLOCKED: Security policy violation")
        mock_hist.filter_by.return_value.order_by.return_value.limit.return_value.all.return_value = [fake_block]

        import collections
        Row = collections.namedtuple('Row', ['ip_address', 'fails'])
        mock_db_query.return_value.filter.return_value.group_by.return_value.order_by.return_value.limit.return_value.all.return_value = [
            Row('192.168.1.5', 42)]

        res = client.get('/api/admin/security-report')
        assert res.status_code == 200
        data = res.get_json()
        assert data['total_spoofing_attempts'] == 12
        assert len(data['recent_system_blocks']) == 1
        assert data['top_suspicious_ips'][0]['ip'] == '192.168.1.5'
        assert data['top_suspicious_ips'][0]['failures'] == 42


def test_endpoints_unauthorized_and_edge_cases(client):
    assert client.get('/api/user/history').status_code == 401
    assert client.post('/api/user/toggle-voice').status_code == 401
    assert client.post('/api/user/reinit_biometrics', json={}).status_code == 401
    assert client.put('/api/user/profile', json={}).status_code == 41

    with client.session_transaction() as sess:
        sess['user_id'] = 999

    with patch('prototype.camera_api.User.query') as mock_query:
        mock_query.get.return_value = None
        assert client.put('/api/user/profile', json={}).status_code == 404
        assert client.post('/api/user/reinit_biometrics', json={}).status_code == 404


def test_update_profile_short_password(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 1

    fake_user = User(id=1, password="old")
    with patch('prototype.camera_api.User.query') as mock_query:
        mock_query.get.return_value = fake_user

        res = client.put('/api/user/profile', json={"password": "short"})
        assert res.status_code == 400
        assert "Hasło musi mieć minimum" in res.get_json()['message']


@patch('prototype.camera_api.base64_to_image')
@patch('prototype.camera_api.detector.get_face')
@patch('prototype.camera_api.facenet.describe')
def test_register_voice_exceptions(mock_desc, mock_face, mock_b64, client):
    mock_b64.return_value = np.zeros((10, 10, 3))
    mock_face.return_value = (np.zeros((10, 10, 3)), (0, 0, 1, 1), None)
    mock_desc.return_value = np.zeros(128)

    with patch('prototype.camera_api.User.query') as mock_query:
        mock_query.filter_by.return_value.first.return_value = None

        payload_no_audio = {
            "first_name": "Test", "last_name": "Test", "email": "a@b.pl",
            "password": "Password123!", "image": "img"
        }

        res1 = client.post('/register', json=payload_no_audio)
        assert res1.status_code == 400
        assert "Voice sample missing" in res1.get_json()['message']

        payload_audio = payload_no_audio.copy()
        payload_audio["audio"] = "bad_audio"
        with patch('prototype.camera_api.base64_to_audio', side_effect=Exception("Decode fail")):
            res2 = client.post('/register', json=payload_audio)
            assert res2.status_code == 400
            assert "Voice processing error" in res2.get_json()['message']


def test_login_check_wrong_credentials(client):
    with patch('prototype.camera_api.User.query') as mock_query, \
            patch('prototype.camera_api.check_password_hash', return_value=False):
        mock_query.filter_by.return_value.first.return_value = User(password="hash")

        res = client.post('/api/login-check', json={"email": "a@b.pl", "password": "złe"})
        assert res.status_code == 400
        assert res.get_json()['valid'] is False


def test_login_blocked_account(client):
    fake_user = User(id=1, email="x@x.pl", password="p", is_active=False)
    with patch('prototype.camera_api.User.query') as mock_query, \
            patch('prototype.camera_api.check_password_hash', return_value=True):
        mock_query.filter_by.return_value.first.return_value = fake_user

        res = client.post('/login', json={"email": "x@x.pl", "password": "p"})
        assert res.status_code == 403
        assert "zablokowane przez administratora" in res.get_json()['message']


@patch('prototype.camera_api.check_security_threats')
@patch('prototype.camera_api.camera.get_frame')
@patch('prototype.camera_api.detector.get_face')
@patch('prototype.camera_api.is_face_distance_valid')
@patch('prototype.camera_api.parse_model_name')
@patch('prototype.camera_api.image_cropper.crop')
@patch('prototype.camera_api.anti_spoof_engine.predict')
@patch('prototype.camera_api.facenet.describe')
@patch('prototype.camera_api.decrypt_embedding')
@patch('prototype.camera_api.base64_to_audio')
@patch('prototype.camera_api.voice_extractor.describe')
def test_login_mfa_voice_and_admin(mock_voice_desc, mock_b64_audio, mock_decrypt, mock_face_desc, mock_predict,
                                   mock_crop, mock_parse, mock_dist, mock_get_face, mock_get_frame, mock_security, client):
    fake_user = User(id=1, email="mfa@test.pl", password="hash", is_active=True, require_voice_auth=True)
    mock_get_frame.return_value = np.zeros((10, 10, 3))
    mock_get_face.return_value = (np.zeros((10, 10, 3)), (0, 0, 10, 10), None)
    mock_dist.return_value = (True, 0.2)
    mock_parse.return_value = (80, 80, "name", 1.0)
    mock_crop.return_value = np.zeros((80, 80, 3))
    mock_predict.return_value = np.array([[0.0, 1.0, 0.0]])
    mock_face_desc.return_value = np.array([0.1, 0.2])
    mock_decrypt.return_value = np.array([0.1, 0.2])

    with patch('prototype.camera_api.User.query') as mock_query, \
            patch('prototype.camera_api.check_password_hash', return_value=True), \
            patch('os.listdir', return_value=['m.pth']):
        mock_query.filter_by.return_value.first.return_value = fake_user

        res_missing = client.post('/login', json={"email": "mfa@test.pl", "password": "p"})
        assert res_missing.status_code == 400

        mock_b64_audio.return_value = b"audio"
        mock_voice_desc.return_value = np.array([0.0, 1.0])
        mock_decrypt.side_effect = [np.array([0.1, 0.2]), np.array([1.0, 0.0])]

        res_mismatch = client.post('/login', json={"email": "mfa@test.pl", "password": "p", "audio": "b64"})
        assert res_mismatch.status_code == 401

        mock_decrypt.side_effect = [np.array([0.1, 0.2]), np.array([1.0, 0.0])]
        mock_voice_desc.side_effect = Exception("Crash")
        res_crash = client.post('/login', json={"email": "mfa@test.pl", "password": "p", "audio": "b64"})
        assert res_crash.status_code == 500

        fake_user.is_admin = True
        mock_voice_desc.side_effect = None
        mock_voice_desc.return_value = np.array([1.0, 0.0])
        mock_decrypt.side_effect = [np.array([0.1, 0.2]), np.array([1.0, 0.0])]

        with patch('prototype.camera_api.db.session.commit', side_effect=Exception("DB Error")):
            res_success = client.post('/login', json={"email": "mfa@test.pl", "password": "p", "audio": "b64"})
            assert res_success.status_code == 200
            assert "Admin login successful" in res_success.get_json()['message']


@patch('prototype.camera_api.base64_to_cv2_img')
@patch('prototype.camera_api.detector.get_face')
def test_reinit_biometrics_edge_cases(mock_get_face, mock_cv2, client):
    with client.session_transaction() as sess:
        sess['user_id'] = 1

    with patch('prototype.camera_api.User.query') as mock_query:
        mock_query.get.return_value = User(id=1)

        mock_cv2.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_get_face.return_value = (None, None, "No Face")
        res1 = client.post('/api/user/reinit_biometrics', json={"image": "img"})
        assert res1.status_code == 400
        assert "Nie wykryto twarzy" in res1.get_json()['message']

        mock_cv2.return_value = None
        res2 = client.post('/api/user/reinit_biometrics', json={"image": "img"})
        assert res2.status_code == 400
        assert "Błąd przetwarzania pliku" in res2.get_json()['message']

        res3 = client.post('/api/user/reinit_biometrics', json={})
        assert res3.status_code == 400
        assert "Nie otrzymano danych" in res3.get_json()['message']

        mock_cv2.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_get_face.return_value = (np.zeros((10, 10, 3)), (0, 0, 1, 1), None)
        with patch('prototype.camera_api.facenet.describe', return_value=np.zeros(128)), \
                patch('prototype.camera_api.db.session.commit', side_effect=Exception("DB fail")):
            res4 = client.post('/api/user/reinit_biometrics', json={"image": "img"})
            assert res4.status_code == 500


@patch('prototype.camera_api.User.query')
def test_admin_required_decorator(mock_query, client):
    assert client.get('/admin/panel').status_code == 401

    mock_query.get.return_value = User(id=1, is_admin=False)
    with client.session_transaction() as sess:
        sess['user_id'] = 1
    assert client.get('/admin/panel').status_code == 403


@patch('prototype.camera_api.User.query')
def test_admin_exceptions_and_empty_config(mock_query, client):
    fake_admin = User(id=99, is_admin=True)
    mock_query.get.return_value = fake_admin
    mock_query.get_or_404.return_value = User(id=2)

    with client.session_transaction() as sess:
        sess['user_id'] = 99

    with patch('prototype.camera_api.db.session.delete', side_effect=Exception("Lock")):
        res1 = client.delete('/api/admin/users/2')
        assert res1.status_code == 500

    with patch('prototype.camera_api.SystemConfig.query') as mock_sys, \
            patch('prototype.camera_api.db.session.add') as mock_add, \
            patch('prototype.camera_api.db.session.commit'):
        mock_sys.first.return_value = None
        client.post('/api/admin/config', json={"face_threshold": 0.5})
        assert mock_add.called

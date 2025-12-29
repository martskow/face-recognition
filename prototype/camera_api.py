import os
import json
import numpy as np
import cv2
import base64
import warnings
from io import BytesIO
from PIL import Image

from flask import Flask, Response, jsonify, render_template, request, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from cryptography.fernet import Fernet

from camera_capture import ImageAcquisition
from FaceDetector import FaceDetector
from FaceNetExtractor import FaceNetExtractor
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'TAI_SESSION_KEY_123'

MODEL_DIR = "prototype/resources/anti_spoof_models"
DEVICE_ID = 0

# --- DB ---
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    embedding_encrypted = db.Column(db.Text, nullable=False)


with app.app_context():
    db.create_all()

# --- Security ---
ENCRYPTION_KEY = b'QUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUE='
cipher_suite = Fernet(ENCRYPTION_KEY)

# --- Init ---
camera = ImageAcquisition(camera_id=0, frame_size=(640, 480))
detector = FaceDetector(dimension=160)
facenet = FaceNetExtractor()

anti_spoof_engine = AntiSpoofPredict(DEVICE_ID)
image_cropper = CropImage()

face_detected = False


def encrypt_embedding(embedding):
    return cipher_suite.encrypt(json.dumps(embedding).encode())


def decrypt_embedding(blob):
    return np.array(json.loads(cipher_suite.decrypt(blob).decode()))


def base64_to_image(base64_str):
    img_bytes = base64.b64decode(base64_str.split(',')[1])
    img = Image.open(BytesIO(img_bytes)).convert('RGB')
    return np.array(img) / 255.0


def generate_frames():
    global face_detected

    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

        face, _, _ = detector.get_face(frame)

        if face is not None:
            frame_bgr = cv2.cvtColor((face * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            face_detected = True
        else:
            frame_bgr = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            face_detected = False

        _, buffer = cv2.imencode('.jpg', frame_bgr)
        yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +
                buffer.tobytes() +
                b'\r\n'
        )


# --- Views ---
@app.route('/')
def login_view():
    return render_template('login.html')


@app.route('/register_view')
def register_view():
    return render_template('register.html')


@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login_view'))
    return render_template('dashboard.html', user=session)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_view'))


# --- API ---
@app.route('/video')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/face_status')
def face_status():
    return jsonify({"face_detected": face_detected})


@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()

    if User.query.filter_by(email=data['email']).first():
        return jsonify({"message": "E-mail already exists"}), 400

    img = base64_to_image(data['image'])
    face, _ = detector.get_face(img)

    if face is None:
        return jsonify({"message": "Face not detected"}), 400

    embedding = facenet.describe(face)
    encrypted_emb = encrypt_embedding(embedding.tolist())

    user = User(
        first_name=data['first_name'],
        last_name=data['last_name'],
        email=data['email'],
        password=generate_password_hash(data['password'], method='pbkdf2:sha256'),
        embedding_encrypted=encrypted_emb
    )

    db.session.add(user)
    db.session.commit()

    return jsonify({"message": "Registration successful!"})


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(email=data.get('email')).first()

    if not user or not check_password_hash(user.password, data.get('password')):
        return jsonify({"message": "Wrong email or password"}), 400

    frame = camera.get_frame()
    if frame is None:
        return jsonify({"message": "Camera access error"}), 500

    frame_cv2 = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    face, coords, _ = detector.get_face(frame)

    if face is None:
        return jsonify({"message": "Face not detected"}), 400

    prediction = np.zeros((1, 3))

    try:
        for model in os.listdir(MODEL_DIR):
            h, w, _, scale = parse_model_name(model)
            params = {
                "org_img": frame_cv2,
                "bbox": coords,
                "scale": scale,
                "out_w": w,
                "out_h": h,
                "crop": scale is not None,
            }

            img_patch = image_cropper.crop(**params)
            prediction += anti_spoof_engine.predict(
                img_patch, os.path.join(MODEL_DIR, model)
            )

        label = np.argmax(prediction)
        score = prediction[0][label] / 2

        if label != 1:
            return jsonify({
                "message": "Spoofing detected",
                "score": float(score)
            }), 403

    except Exception as e:
        return jsonify({"message": f"Anti-spoofing error: {e}"}), 500

    embedding = facenet.describe(face)
    stored = decrypt_embedding(user.embedding_encrypted)

    if np.linalg.norm(embedding - stored) > 0.6:
        return jsonify({"message": "Face does not match biometrics"}), 401

    session.update({
        "user_id": user.id,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "email": user.email
    })

    return jsonify({"message": "Login successful", "redirect": url_for('dashboard')})


if __name__ == "__main__":
    try:
        app.run(host='0.0.0.0', port=697, debug=True)
    finally:
        camera.release()

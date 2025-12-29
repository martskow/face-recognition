import os
import json
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
from flask import Flask, Response, jsonify, render_template, request, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from cryptography.fernet import Fernet

# Importy Twoich modułów
from camera_capture import ImageAcquisition
from FaceDetector import FaceDetector
from FaceNetExtractor import FaceNetExtractor

app = Flask(__name__)
app.secret_key = 'TAI_SESSION_KEY_123'

# --- KONFIGURACJA BAZY DANYCH ---
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- MODEL BAZY DANYCH ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    embedding_encrypted = db.Column(db.Text, nullable=False)

# Tworzenie pliku bazy danych i tabel
with app.app_context():
    db.create_all()

# --- KONFIGURACJA BEZPIECZEŃSTWA (Klucz bitowy A) ---
ENCRYPTION_KEY = b'QUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUE='
cipher_suite = Fernet(ENCRYPTION_KEY)

# --- INICJALIZACJA ---
camera = ImageAcquisition(camera_id=0, frame_size=(640, 480))
detector = FaceDetector(dimension=160)
facenet = FaceNetExtractor()
face_detected = False

def encrypt_embedding(embedding_list):
    json_data = json.dumps(embedding_list).encode('utf-8')
    return cipher_suite.encrypt(json_data)

def decrypt_embedding(encrypted_blob):
    decrypted_data = cipher_suite.decrypt(encrypted_blob)
    return np.array(json.loads(decrypted_data.decode('utf-8')))

def generate_frames():
    global face_detected
    while True:
        frame = camera.get_frame()
        if frame is None: continue
        face, _ = detector.get_face(frame)
        if face is not None:
            frame_bgr = (face * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
            face_detected = True
        else:
            frame_bgr = (frame * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
            face_detected = False
        ret, buffer = cv2.imencode('.jpg', frame_bgr)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def base64_to_image(base64_str):
    img_bytes = base64.b64decode(base64_str.split(',')[1])
    img = Image.open(BytesIO(img_bytes)).convert('RGB')
    return np.array(img) / 255.0

# --- ROUTY WIDOKÓW ---

@app.route('/')
def login_view():
    return render_template('login.html')

@app.route('/register_view')
def register_view():
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    # Sprawdzenie czy użytkownik jest w sesji
    if 'user_id' not in session:
        return redirect(url_for('login_view'))
    return render_template('dashboard.html', user=session)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_view'))

# --- ROUTY API ---

@app.route('/video')
def video_feed():
    response = Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route("/face_status")
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
    # Używamy pbkdf2:sha256 aby uniknąć błędu scrypt na macOS
    hashed_password = generate_password_hash(data['password'], method='pbkdf2:sha256')
    encrypted_emb = encrypt_embedding(embedding.tolist())

    new_user = User(
        first_name=data['first_name'],
        last_name=data['last_name'],
        email=data['email'],
        password=hashed_password,
        embedding_encrypted=encrypted_emb
    )
    db.session.add(new_user)
    db.session.commit()
    return jsonify({"message": "Registration successful!"})

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(email=data.get('email')).first()

    if not user or not check_password_hash(user.password, data.get('password')):
        return jsonify({"message": "Wrong email or password"}), 400

    frame = camera.get_frame()
    face, _ = detector.get_face(frame)
    if face is None:
        return jsonify({"message": "Face not detected"}), 400

    embedding = facenet.describe(face)
    stored_embedding = decrypt_embedding(user.embedding_encrypted)
    dist = np.linalg.norm(embedding - stored_embedding)

    # Próg 0.6 (zachowany z Twojego ostatniego kodu)
    if dist > 0.6:
        return jsonify({"message": "Face does not match"}), 401

    # ZAPIS DO SESJI
    session['user_id'] = user.id
    session['first_name'] = user.first_name
    session['last_name'] = user.last_name
    session['email'] = user.email

    return jsonify({"message": "Login successful", "redirect": url_for('dashboard')})

if __name__ == "__main__":
    try:
        app.run(host='0.0.0.0', port=697, debug=True)
    finally:
        camera.release()
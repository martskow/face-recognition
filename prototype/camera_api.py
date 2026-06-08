import os
import json
import numpy as np
import cv2
import csv
import warnings
from io import BytesIO, StringIO
from PIL import Image
import re
from datetime import datetime, timedelta
import base64
from sqlalchemy import func
from flask import Flask, Response, jsonify, make_response, render_template, request, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from cryptography.fernet import Fernet

from camera_capture import ImageAcquisition
from FaceDetector import FaceDetector
from FaceNetExtractor import FaceNetExtractor
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

from VoiceExtractor import VoiceEmbeddingExtractor

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
    voice_embedding_encrypted = db.Column(db.Text, nullable=True)

    is_admin = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=True)

    require_voice_auth = db.Column(db.Boolean, default=True)


class LoginHistory(db.Model):
    __tablename__ = 'login_history'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    ip_address = db.Column(db.String(45), nullable=True)
    user_agent = db.Column(db.String(255), nullable=True)
    status = db.Column(db.String(50),
                       nullable=False)  # np. "Success", "Failed: Face mismatch", "Failed: Voice mismatch"

    # Opcjonalnie relacja, żeby łatwo wyciągać dane użytkownika
    user = db.relationship('User', backref=db.backref('login_histories', lazy=True))

# =====================================================================
# 1. NOWY MODEL BAZY DANYCH DLA PARAMETRÓW GLOBALNYCH
# =====================================================================
class SystemConfig(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    face_threshold = db.Column(db.Float, default=0.6)  # Domyślny próg FaceNet (Euklidesowa)
    voice_threshold = db.Column(db.Float, default=0.45)  # Domyślny próg głosu (Cosinusowa)

with app.app_context():
    db.create_all()

# --- Security ---
ENCRYPTION_KEY = b'QUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUE='
cipher_suite = Fernet(ENCRYPTION_KEY)

# --- Init ---
camera = ImageAcquisition(camera_id=0, frame_size=(640, 480))
detector = FaceDetector(dimension=160)
facenet = FaceNetExtractor()
voice_extractor = VoiceEmbeddingExtractor()

anti_spoof_engine = AntiSpoofPredict(DEVICE_ID)
image_cropper = CropImage()

face_detected = False

MIN_FACE_AREA_RATIO = 0.08
MAX_FACE_AREA_RATIO = 0.35


def encrypt_embedding(embedding):
    return cipher_suite.encrypt(json.dumps(embedding).encode())


def decrypt_embedding(blob):
    return np.array(json.loads(cipher_suite.decrypt(blob).decode()))


def base64_to_image(base64_str):
    img_bytes = base64.b64decode(base64_str.split(',')[1])
    img = Image.open(BytesIO(img_bytes)).convert('RGB')
    return np.array(img) / 255.0


def base64_to_audio(base64_str):
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    return base64.b64decode(base64_str)


def is_face_distance_valid(bbox, frame_shape):
    x, y, w, h = bbox
    frame_h, frame_w = frame_shape[:2]

    face_area = w * h
    frame_area = frame_w * frame_h

    ratio = face_area / frame_area

    return MIN_FACE_AREA_RATIO <= ratio <= MAX_FACE_AREA_RATIO, ratio


def validate_user_data(data):
    errors = []

    name_regex = r"^[A-Za-zżźćńółęąśŻŹĆŃÓŁĘĄŚ\-]{2,50}$"

    if not re.match(name_regex, data.get('first_name', '')):
        errors.append("Name contains forbidden characters or is too short (2 characters minimum)")

    if not re.match(name_regex, data.get('last_name', '')):
        errors.append("Surname contains forbidden characters or is too short (2 characters minimum)")

    email_regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    if not re.match(email_regex, data.get('email', '')):
        errors.append("Invalid email address syntax")

    if len(data.get('password', '')) < 8:
        errors.append("Password has to be at least 8 characters long")

    return errors


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


def check_security_threats(user):
    time_threshold = datetime.utcnow() - timedelta(minutes=15)

    recent_logs = LoginHistory.query.filter(
        LoginHistory.user_id == user.id,
        LoginHistory.timestamp >= time_threshold
    ).all()

    failed_logins = 0
    spoofing_attempts = 0

    for log in recent_logs:
        if log.status.startswith("Failed"):
            failed_logins += 1
        if "Spoofing detected" in log.status:
            spoofing_attempts += 1

    if failed_logins >= 5 or spoofing_attempts >= 3:
        user.is_active = False
        db.session.commit()

        lockdown_log = LoginHistory(
            user_id=user.id,
            ip_address="SYSTEM",
            user_agent="SYSTEM_AUTO_LOCK",
            status="BLOCKED: Security policy violation (Brute-force / Spoofing)"
        )
        db.session.add(lockdown_log)
        db.session.commit()
        print(f"⚠️ KONTO ZABLOKOWANE AUTOMATYCZNIE: {user.email}")

# --- Views ---
@app.route('/')
def login_view():
    return render_template('login.html')


@app.route('/register_view')
def register_view():
    return render_template('register.html')


@app.route('/dashboard')
def dashboard():
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('login'))

    current_user = User.query.get(user_id)

    return render_template('dashboard.html', user=current_user)


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
    validation_errors = validate_user_data(data)
    if validation_errors:
        return jsonify({"message": validation_errors[0]}), 400
    if User.query.filter_by(email=data['email']).first():
        return jsonify({"message": "E-mail already exists"}), 400

    img = base64_to_image(data['image'])
    face, coords, _ = detector.get_face(img)

    if face is None:
        return jsonify({"message": "Face not detected"}), 400

    embedding = facenet.describe(face)
    encrypted_emb = encrypt_embedding(embedding.tolist())

    if 'audio' not in data:
        return jsonify({"message": "Voice sample missing"}), 400

    try:
        audio_bytes = base64_to_audio(data['audio'])
        voice_embedding = voice_extractor.describe(audio_bytes)
        encrypted_voice_emb = encrypt_embedding(voice_embedding.tolist())
    except Exception as e:
        return jsonify({"message": f"Voice processing error: {e}"}), 400

    user = User(
        first_name=data['first_name'],
        last_name=data['last_name'],
        email=data['email'],
        password=generate_password_hash(data['password'], method='pbkdf2:sha256'),
        embedding_encrypted=encrypted_emb,
        voice_embedding_encrypted=encrypted_voice_emb  # Zapis do bazy danych
    )

    db.session.add(user)
    db.session.commit()

    return jsonify({"message": "Registration successful!"})


@app.route('/api/login-check', methods=['POST'])
def login_check():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    user = User.query.filter_by(email=email).first()

    # Podstawowa weryfikacja danych, żeby nie zdradzać konfiguracji losowym osobom
    if not user or not check_password_hash(user.password, password):
        return jsonify({"valid": False, "message": "Wrong email or password"}), 400

    # Sprawdzamy też blokadę konta z panelu admina
    if not getattr(user, 'is_active', True):
        return jsonify(
            {"valid": False, "message": "Twoje konto zostało zablokowane przez administratora systemu!"}), 403

    return jsonify({
        "valid": True,
        "require_voice": getattr(user, 'require_voice_auth', True)
    })


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    user = User.query.filter_by(email=email).first()

    # Pomocnicza funkcja do szybkiego zapisywania nieudanych prób
    def log_failed_attempt(user_obj, error_msg):
        if user_obj: # Logujemy tylko, jeśli użytkownik istnieje w bazie
            try:
                failed_log = LoginHistory(
                    user_id=user_obj.id,
                    ip_address=request.remote_addr,
                    user_agent=request.user_agent.string[:255],
                    status=f"Failed ({error_msg})"
                )
                db.session.add(failed_log)
                db.session.commit()

                check_security_threats(user_obj)
            except Exception as e:
                print(f"Błąd zapisu historii logowania: {e}")

    # 1. BŁĄD: Zły mail lub hasło (Kod 400)
    if not user or not check_password_hash(user.password, data.get('password')):
        log_failed_attempt(user, "400: Wrong credentials")
        return jsonify({"message": "Wrong email or password"}), 400

    # === DODATEK ADMINA 1: KONTROLA BLOKADY KONTA ===
    if not getattr(user, 'is_active', True):
        log_failed_attempt(user, "403: Account suspended by administrator")
        return jsonify({"message": "Twoje konto zostalo zablokowane przez administratora systemu!"}), 403

    # === DODATEK ADMINA 2: DYNAMICZNE POBIERANIE PROGÓW CZUŁOŚCI ===
    config = SystemConfig.query.first()
    face_th = config.face_threshold if config else 0.6
    voice_th = config.voice_threshold if config else 0.45

    frame = camera.get_frame()
    if frame is None:
        log_failed_attempt(user, "500: Camera error")
        return jsonify({"message": "Camera access error"}), 500

    frame_cv2 = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    face, coords, _ = detector.get_face(frame)

    # 2. BŁĄD: Nie wykryto twarzy (Kod 400)
    if face is None:
        log_failed_attempt(user, "400: Face not detected")
        return jsonify({"message": "Face not detected"}), 400

    valid, ratio = is_face_distance_valid(coords, frame.shape)

    # 3. BŁĄD: Twarz za blisko / za daleko (Kod 400)
    if not valid:
        if ratio > MAX_FACE_AREA_RATIO:
            log_failed_attempt(user, "400: Face too close")
            return jsonify({"message": "Face too close to camera"}), 400
        else:
            log_failed_attempt(user, "400: Face too far")
            return jsonify({"message": "Face too far from camera"}), 400

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

        # 4. BŁĄD: Atak prezentacji / Spoofing (Kod 403)
        if label != 1:
            log_failed_attempt(user, "403: Spoofing detected")
            return jsonify({"message": "Spoofing detected", "score": float(score)}), 403

    except Exception as e:
        log_failed_attempt(user, "500: Anti-spoofing crash")
        return jsonify({"message": f"Anti-spoofing error: {e}"}), 500

    embedding = facenet.describe(face)
    stored = decrypt_embedding(user.embedding_encrypted)

    # 5. BŁĄD: Niedopasowanie biometrii twarzy (Używa zmiennej dynamicznej face_th)
    if np.linalg.norm(embedding - stored) > face_th:
        log_failed_attempt(user, "401: Face mismatch")
        return jsonify({"message": "Face does not match biometrics"}), 401

    if getattr(user, 'require_voice_auth', True):

        # 6. BŁĄD: Brak przesłanego audio (Kod 400)
        if 'audio' not in data or not data['audio']:
            log_failed_attempt(user, "400: Audio missing")
            return jsonify({"message": "Voice verification required"}), 400

        try:
            audio_bytes = base64_to_audio(data['audio'])
            current_voice_emb = voice_extractor.describe(audio_bytes)
            stored_voice_emb = decrypt_embedding(user.voice_embedding_encrypted)

            dot_product = np.dot(current_voice_emb, stored_voice_emb)
            norm_current = np.linalg.norm(current_voice_emb)
            norm_stored = np.linalg.norm(stored_voice_emb)

            cosine_similarity = dot_product / (norm_current * norm_stored)

            # 7. BŁĄD: Niedopasowanie głosu
            if cosine_similarity < voice_th:
                log_failed_attempt(user, "401: Voice mismatch")
                return jsonify({"message": "Voice does not match biometrics"}), 401

        except Exception as e:
            log_failed_attempt(user, "500: Voice engine crash")
            return jsonify({"message": f"Voice verification error: {e}"}), 500

    else:
        print(f"ℹUżytkownik {user.email} ma wyłączone MFA. Pomijam weryfikację głosu.")

    # === SUKCES: UDANE LOGOWANIE ===
    try:
        success_log = LoginHistory(
            user_id=user.id,
            ip_address=request.remote_addr,
            status="Passed"
        )
        db.session.add(success_log)
        db.session.commit()
    except Exception as e:
        print(f"Błąd zapisu historii logowania: {e}")

    session.update({
        "user_id": user.id,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "email": user.email
    })

    if user.is_admin:
        return jsonify({"message": "Admin login successful", "redirect": url_for('admin_panel')})
    else:
        # Zwykły użytkownik ląduje na standardowym dashboardzie
        return jsonify({"message": "Login successful", "redirect": url_for('dashboard')})




# ENDPOINTY DLA Z3 (UŻYTKOWNIK)

@app.route('/api/user/profile', methods=['PUT'])
def update_profile():
    # Pobieramy ID zalogowanego użytkownika z sesji Flaska
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"message": "Brak autoryzacji. Zaloguj się ponownie."}), 41

    data = request.get_json()
    user = User.query.get(user_id)

    if not user:
        return jsonify({"message": "Użytkownik nie istnieje."}), 404

    # Aktualizacja danych podstawowych
    if data.get('first_name'):
        user.first_name = data.get('first_name')
    if data.get('last_name'):
        user.last_name = data.get('last_name')

    # Aktualizacja hasła
    password = data.get('password')
    if password and len(password) >= 8:
        from werkzeug.security import generate_password_hash
        user.password = generate_password_hash(password)
    elif password and len(password) < 8:
        return jsonify({"message": "Hasło musi mieć minimum 8 znaków!"}), 400

    db.session.commit()
    return jsonify({"message": "Profil zaktualizowany pomyślnie!"})


@app.route('/api/user/history', methods=['GET'])
def get_user_history():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"message": "Brak autoryzacji."}), 401

    history_records = LoginHistory.query.filter_by(user_id=user_id).order_by(LoginHistory.timestamp.desc()).all()

    history_list = []
    for record in history_records:
        history_list.append({
            # formatowanie daty do czytelnego stringa
            "timestamp": record.timestamp.strftime('%Y-%m-%d %H:%M:%S') if isinstance(record.timestamp,
                                                                                      datetime) else str(
                record.timestamp),
            "ip": record.ip_address if hasattr(record, 'ip_address') else getattr(record, 'ip', 'Nieznane'),
            "status": record.status
        })

    return jsonify(history_list)


def base64_to_cv2_img(base64_string):
    try:
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Błąd dekodowania obrazu Base64: {e}")
        return None


@app.route('/api/user/reinit_biometrics', methods=['POST'])
def reinit_biometrics():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"message": "Brak autoryzacji. Zaloguj się ponownie."}), 401

    data = request.get_json()
    user = User.query.get(user_id)

    if not user:
        return jsonify({"message": "Użytkownik nie istnieje."}), 404

    image_base64 = data.get('image')
    audio_base64 = data.get('audio')

    updated_parts = []

    try:
        # === 1. AKTUALIZACJA TWARZY ===
        if image_base64:
            frame_cv2 = base64_to_cv2_img(image_base64)
            if frame_cv2 is not None:
                frame_rgb = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB)

                # Pobranie wyciętej twarzy z detektora
                face, coords, _ = detector.get_face(frame_rgb)
                if face is not None:
                    # Generowanie wektora (zwraca ndarray z FaceNetExtractor)
                    new_embedding = facenet.describe(face)

                    # !!! KLUCZOWE ZABEZPIECZENIE !!!
                    # Konwertujemy ndarray na standardową listę Pythona, aby zapobiec błędowi serializacji
                    if isinstance(new_embedding, np.ndarray):
                        new_embedding = new_embedding.tolist()

                    # Szyfrujemy już czystą listę / bezpieczny obiekt
                    user.embedding_encrypted = encrypt_embedding(new_embedding)
                    updated_parts.append("Twarz")
                else:
                    return jsonify({"message": "Nie wykryto twarzy na przesłanym zdjęciu. Ustaw się prosto."}), 400
            else:
                return jsonify({"message": "Błąd przetwarzania pliku graficznego."}), 400

        # === 2. AKTUALIZACJA GŁOSU ===
        if audio_base64:
            audio_bytes = base64_to_audio(audio_base64)
            new_voice_emb = voice_extractor.describe(audio_bytes)

            # !!! KLUCZOWE ZABEZPIECZENIE DLA GŁOSU !!!
            if isinstance(new_voice_emb, np.ndarray):
                new_voice_emb = new_voice_emb.tolist()

            user.voice_embedding_encrypted = encrypt_embedding(new_voice_emb)
            updated_parts.append("Glos")

        # === 3. WALIDACJA I ZAPIS ZDARZENIA ===
        if not updated_parts:
            return jsonify({"message": "Nie otrzymano danych biometrycznych."}), 400

        status_msg = f"Aktualizacja biometrii ({', '.join(updated_parts)})"

        biometric_log = LoginHistory(
            user_id=user.id,
            ip_address=request.remote_addr,
            status=status_msg
        )

        db.session.add(biometric_log)
        db.session.commit()

        # Przekazujemy czysty string w odpowiedzi
        return jsonify({"message": f"Pomyślnie zaktualizowano wzorce biometryczne: {str(status_msg)}"})

    except Exception as e:
        db.session.rollback()
        return jsonify({"message": f"Błąd aktualizacji biometrii: {str(e)}"}), 500


from flask import render_template, abort



# =====================================================================
# 2. DEKORATOR ZABEZPIECZAJĄCY (Weryfikacja czy zalogowany to admin)
# =====================================================================
def admin_required():
    user_id = session.get('user_id')
    if not user_id:
        abort(401)  # Brak autoryzacji
    user = User.query.get(user_id)
    if not user or not getattr(user, 'is_admin', False):
        abort(403)  # Zabroniony dostęp
    return user


# =====================================================================
# 3. ENDPOINTY PANELU ADMINISTRATORA
# =====================================================================

@app.route('/admin/panel')
def admin_panel():
    admin_required()  # Sprawdzenie uprawnień
    # Pobranie obecnej konfiguracji lub utworzenie domyślnej
    config = SystemConfig.query.first()
    if not config:
        config = SystemConfig(face_threshold=0.6, voice_threshold=0.45)
        db.session.add(config)
        db.session.commit()
    return render_template('admin.html', config=config)


# API: Pobieranie wszystkich kont
@app.route('/api/admin/users', methods=['GET'])
def admin_get_users():
    admin_required()
    users = User.query.all()
    users_list = []
    for u in users:
        users_list.append({
            "id": u.id,
            "first_name": u.first_name,
            "last_name": u.last_name,
            "email": u.email,
            "is_active": getattr(u, 'is_active', True),
            "is_admin": u.is_admin
        })
    return jsonify(users_list)


# API: Blokowanie / Aktywowanie użytkownika
@app.route('/api/admin/users/<int:target_id>/toggle', methods=['POST'])
def admin_toggle_user(target_id):
    admin_required()
    user = User.query.get_or_404(target_id)

    if user.id == session.get('user_id'):
        return jsonify({"message": "Nie możesz zablokować samego siebie!"}), 400

    user.is_active = not getattr(user, 'is_active', True)
    db.session.commit()

    stan = "odblokowane" if user.is_active else "zablokowane"
    return jsonify({"message": f"Konto użytkownika {user.email} zostało {stan}."})


# API: Trwałe usuwanie użytkownika
@app.route('/api/admin/users/<int:target_id>', methods=['DELETE'])
def admin_delete_user(target_id):
    admin_required()
    user = User.query.get_or_404(target_id)

    if user.id == session.get('user_id'):
        return jsonify({"message": "Nie możesz usunąć swojego własnego konta!"}), 400

    try:
        # 1. RĘCZNIE CZYŚCIMY HISTORIĘ LOGOWANIA TEGO UŻYTKOWNIKA
        # (Podmień 'LoginHistory' na dokładną nazwę swojej klasy historii, jeśli jest inna)
        LoginHistory.query.filter_by(user_id=user.id).delete()

        # 2. TERAZ USUWAMY SAMEGO UŻYTKOWNIKA
        db.session.delete(user)
        db.session.commit()

        return jsonify({"message": f"Użytkownik {user.email} oraz jego historia zostali pomyślnie usunięci."})

    except Exception as e:
        db.session.rollback()
        return jsonify({"message": f"Błąd podczas usuwania: {str(e)}"}), 500

# API: Aktualizacja progów czułości algorytmów
@app.route('/api/admin/config', methods=['POST'])
def admin_update_config():
    admin_required()
    data = request.get_json()

    config = SystemConfig.query.first()
    if not config:
        config = SystemConfig()
        db.session.add(config)

    config.face_threshold = float(data.get('face_threshold', 0.6))
    config.voice_threshold = float(data.get('voice_threshold', 0.45))
    db.session.commit()

    return jsonify({"message": "Globalne parametry biometrii zostały pomyślnie zaktualizowane!"})


@app.route('/api/user/toggle-voice', methods=['POST'])
def toggle_voice_auth():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"message": "Niezalogowany"}), 401

    user = User.query.get(user_id)
    user.require_voice_auth = not getattr(user, 'require_voice_auth', True)
    db.session.commit()

    status = "włączona" if user.require_voice_auth else "wyłączona"
    return jsonify({
        "message": f"Weryfikacja głosem została {status}.",
        "require_voice": user.require_voice_auth
    })


@app.route('/api/admin/security-report', methods=['GET'])
def admin_security_report():
    admin_required()

    spoofing_count = LoginHistory.query.filter(LoginHistory.status.like('%Spoofing%')).count()

    recent_blocks = LoginHistory.query.filter_by(
        status='BLOCKED: Security policy violation (Brute-force / Spoofing)').order_by(
        LoginHistory.timestamp.desc()).limit(10).all()

    blocks_data = [{
        "user_id": b.user_id,
        "date": b.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        "reason": b.status
    } for b in recent_blocks]

    top_suspect_ips = db.session.query(
        LoginHistory.ip_address, func.count(LoginHistory.id).label('fails')
    ).filter(LoginHistory.status.like('Failed%')).group_by(LoginHistory.ip_address).order_by(
        func.count(LoginHistory.id).desc()).limit(5).all()

    ip_data = [{"ip": row.ip_address, "failures": row.fails} for row in top_suspect_ips]

    return jsonify({
        "total_spoofing_attempts": spoofing_count,
        "recent_system_blocks": blocks_data,
        "top_suspicious_ips": ip_data
    })


@app.route('/api/admin/export-report', methods=['GET'])
def admin_export_report():
    admin_required()

    logs = LoginHistory.query.order_by(LoginHistory.timestamp.desc()).all()

    si = StringIO()
    si.write('\ufeff')

    cw = csv.writer(si, delimiter=';')

    cw.writerow(
        ['ID_Zdarzenia', 'ID_Uzytkownika', 'Data_i_Czas', 'Adres_IP', 'Urzadzenie_UserAgent', 'Status_Logowania'])

    for log in logs:
        cw.writerow([
            log.id,
            log.user_id,
            log.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            log.ip_address,
            log.user_agent,
            log.status
        ])

    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=raport_bezpieczenstwa_audyt.csv"
    output.headers["Content-type"] = "text/csv"

    return output


if __name__ == "__main__":
    try: # pragma: no cover
        app.run(host='0.0.0.0', port=697, debug=True, use_reloader = False) # pragma: no cover
    finally: # pragma: no cover
        camera.release() # pragma: no cover
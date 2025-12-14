from flask import Flask, Response, jsonify, render_template
from camera_capture import ImageAcquisition
import cv2
import numpy as np
from FaceDetector import FaceDetector
from FaceNetExtractor import FaceNetExtractor
from flask import request
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# "Baza danych" w pamięci na potrzeby prototypu
users_db = {}

# Inicjalizacja kamery
camera = ImageAcquisition(camera_id=0, frame_size=(640, 480))
detector = FaceDetector(dimension=160)
facenet = FaceNetExtractor()

def generate_frames():
    """Generator zwracający strumień wideo w formacie MJPEG z detekcją twarzy"""
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

        # Wykrywanie twarzy
        face, warning = detector.get_face(frame)

        if face is not None:
            # RGB [0,1] -> BGR uint8 [0,255] do OpenCV
            frame_bgr = (face * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
        else:
            # jeśli brak twarzy, pokazujemy oryginalną klatkę
            frame_bgr = (frame * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)

        # do JPEG
        ret, buffer = cv2.imencode('.jpg', frame_bgr)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def base64_to_image(base64_str):
    img_bytes = base64.b64decode(base64_str.split(',')[1])
    img = Image.open(BytesIO(img_bytes))
    img = img.convert('RGB')
    return np.array(img) / 255.0  # normalizacja do [0,1]

@app.route('/video')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/frame')
def single_frame():
    """Zwraca wyciętą twarz i embedding w JSON (jeśli wykryta)"""
    frame = camera.get_frame()
    if frame is None:
        return jsonify({"error": "Unable to download frame"}), 500

    face, warning = detector.get_face(frame)
    if face is None:
        return jsonify({"error": "No face detected"}), 404

    embedding = facenet.describe(face)

    return jsonify({
        "face": face.tolist(),
        "embedding": embedding.tolist(),
        "warning": warning
    })


@app.route('/')
def login_view():
    return render_template('login.html')

@app.route('/register_view')
def register_view():
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    required_fields = ['first_name','last_name','email','password','password2','image']
    if not all(field in data and data[field] for field in required_fields):
        return jsonify({"message": "Fill all required fields"}), 400

    if data['password'] != data['password2']:
        return jsonify({"message": "Passwords are different"}), 400

    if data['email'] in users_db:
        return jsonify({"message": "E-mail is already register in database"}), 400

    img = base64_to_image(data['image'])

    face, warning = detector.get_face(img)
    if face is None:
        return jsonify({"message": "Face was not detected, stand in front of the camera and try again"}), 400

    embedding = facenet.describe(face)

    # TODO: zapisz użytkownika i embedding do prawdziwej bazy danych
    users_db[data['email']] = {
        "first_name": data['first_name'],
        "last_name": data['last_name'],
        "password": data['password'],
        "embedding": embedding.tolist()
    }

    return jsonify({"message": f"User {data['first_name']} {data['last_name']} registered!"})

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not all([email, password]):
        return jsonify({"message": "Fill email and password"}), 400

    if email not in users_db:
        return jsonify({"message": "User not found"}), 404

    user = users_db[email]
    if password != user['password']:
        return jsonify({"message": "Wrong password"}), 401

    frame = camera.get_frame()
    if frame is None:
        return jsonify({"message": "Camera error"}), 500

    face, _ = detector.get_face(frame)
    if face is None:
        return jsonify({"message": "Face not detected"}), 400

    embedding = facenet.describe(face)

    # TODO: zweryfikować prób i obliczanie odległości
    stored = np.array(user['embedding'])
    dist = np.linalg.norm(embedding - stored)

    if dist > 0.9:
        return jsonify({"message": "Face does not match"}), 401

    return jsonify({"message": "Login successful"})

if __name__ == "__main__":
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        camera.release()
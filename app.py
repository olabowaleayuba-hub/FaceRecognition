import os
import sys
import cv2
import numpy as np
import subprocess
import atexit
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACES_DIR = os.path.join(BASE_DIR, 'faces')
TRAINER_FILE = os.path.join(BASE_DIR, 'trainer.yml')

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

recognizer = None
label_map = {}


def load_recognizer():
    global recognizer, label_map
    if os.path.exists(TRAINER_FILE):
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(TRAINER_FILE)
            if os.path.exists(FACES_DIR):
                names = sorted([d for d in os.listdir(FACES_DIR) if os.path.isdir(os.path.join(FACES_DIR, d))])
                label_map = {idx: name for idx, name in enumerate(names)}
        except Exception as e:
            print(f"Error loading recognizer: {e}")


load_recognizer()

# Global State Management
current_mode = 'idle'
target_name = ''
sample_count = 0
MAX_SAMPLES = 100

# Camera Hardware State Management
camera = None
is_camera_active = False

detection_logs = []
last_logged_time = {}


def get_camera():
    """Initializes and returns the camera if not already active."""
    global camera, is_camera_active
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
    is_camera_active = True
    return camera


def release_camera():
    """Completely releases the physical webcam hardware and turns off the light."""
    global camera, is_camera_active
    is_camera_active = False
    if camera is not None:
        camera.release()
        cv2.destroyAllWindows()
        camera = None


def generate_frames():
    global current_mode, target_name, sample_count, recognizer, label_map, detection_logs, last_logged_time, is_camera_active

    cap = get_camera()

    # Loop ONLY runs while camera state is active
    while is_camera_active and cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # MODE 1: Collecting Samples
            if current_mode == 'collect' and target_name:
                face_img = gray[y:y + h, x:x + w]
                face_img = cv2.resize(face_img, (200, 200))

                user_folder = os.path.join(FACES_DIR, target_name)
                os.makedirs(user_folder, exist_ok=True)

                sample_count += 1
                img_path = os.path.join(user_folder, f"{sample_count}.jpg")
                cv2.imwrite(img_path, face_img)

                cv2.putText(frame, f"Capturing {target_name}: {sample_count}/{MAX_SAMPLES}",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if sample_count >= MAX_SAMPLES:
                    current_mode = 'idle'
                    target_name = ''
                    sample_count = 0

            # MODE 2: Live Face Recognition & Timestamp Logging
            elif current_mode == 'recognize' and recognizer is not None:
                face_img = gray[y:y + h, x:x + w]
                face_img = cv2.resize(face_img, (200, 200))
                try:
                    label_id, confidence = recognizer.predict(face_img)
                    if confidence < 75:
                        name = label_map.get(label_id, "Unknown")
                        text = f"{name} ({round(100 - confidence)}%)"
                        color = (0, 255, 0)
                    else:
                        name = "Unknown"
                        text = "Unknown"
                        color = (0, 0, 255)
                except Exception:
                    name = "Unknown"
                    text = "Unknown"
                    color = (0, 0, 255)

                cv2.putText(frame, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Log Name & Time (5-second cooldown per person)
                now_sec = time.time()
                if name not in last_logged_time or (now_sec - last_logged_time[name]) > 5:
                    time_str = datetime.now().strftime("%I:%M:%S %p")
                    detection_logs.insert(0, {"name": name, "time": time_str})
                    last_logged_time[name] = now_sec

            # MODE 3: Default Idle Stream
            elif current_mode == 'idle':
                cv2.putText(frame, "Face Detected", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def cleanup():
    release_camera()


atexit.register(cleanup)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/collect', methods=['POST'])
def collect_faces():
    global current_mode, target_name, sample_count
    data = request.get_json() or {}
    name = data.get('name', '').strip()
    if not name:
        return jsonify({"status": "error", "message": "Please enter a name!"}), 400

    target_name = name
    sample_count = 0
    current_mode = 'collect'
    return jsonify({"status": "success", "message": f"Capturing 100 samples for '{name}'..."})


@app.route('/api/train', methods=['POST'])
def train_model():
    global current_mode
    current_mode = 'idle'

    script_path = os.path.join(BASE_DIR, 'train_model.py')
    if not os.path.exists(script_path):
        script_path = os.path.join(BASE_DIR, 'trainer', 'train_model.py')

    if os.path.exists(script_path):
        subprocess.run([sys.executable, script_path])

    load_recognizer()
    return jsonify({"status": "success", "message": "Model trained & loaded successfully!"})


@app.route('/api/recognize', methods=['POST'])
def recognize():
    global current_mode
    load_recognizer()
    if recognizer is None:
        return jsonify({"status": "error", "message": "No trained model found. Please train model first!"})

    current_mode = 'recognize'
    return jsonify({"status": "success", "message": "Live face recognition active!"})


# TURN OFF CAMERA HARDWARE ENDPOINT
@app.route('/api/stop', methods=['POST'])
def stop_system():
    global current_mode, target_name, sample_count
    current_mode = 'idle'
    target_name = ''
    sample_count = 0

    # RELEASES THE WEBCAM HARDWARE AND TURNS LIGHT OFF
    release_camera()
    return jsonify({"status": "success", "message": "Camera hardware released and powered OFF."})


@app.route('/api/captures', methods=['GET'])
def get_captures():
    return jsonify({
        "status": "success",
        "count": len(detection_logs),
        "logs": detection_logs[:15]
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
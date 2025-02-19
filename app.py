from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json
import time

app = Flask(__name__)

# Load gesture classes
with open("gesture_classes.json", 'r') as f:
    gesture_names = json.load(f)

# Load the trained model
model = tf.keras.models.load_model("flip_hand_gesture_model.h5")

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Global Variables
is_streaming = False
confidence_scores = []

# Function to extract landmarks
def extract_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)
    landmarks = []
    
    if results.left_hand_landmarks:
        for landmark in results.left_hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
    else:
        landmarks.extend([0] * (21 * 3))
    
    if results.right_hand_landmarks:
        for landmark in results.right_hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
    else:
        landmarks.extend([0] * (21 * 3))
    
    return landmarks

# Video streaming
def generate_frames():
    global is_streaming, confidence_scores
    cap = cv2.VideoCapture(0)
    sequence = []
    SEQUENCE_LENGTH = 40
    
    while is_streaming:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            landmarks = extract_landmarks(frame)
            
            if len(sequence) < SEQUENCE_LENGTH:
                sequence.append(landmarks)
            else:
                sequence.pop(0)
                sequence.append(landmarks)
                prediction = model.predict(np.expand_dims(sequence, axis=0), verbose=0)
                class_idx = np.argmax(prediction[0])
                confidence = np.max(prediction[0])
                confidence_scores.append(confidence)
                
                if len(confidence_scores) > 50:
                    confidence_scores.pop(0)
                
                gesture = gesture_names[class_idx] if confidence > 0.8 else "Unknown"
                cv2.putText(frame, f"Prediction: {gesture}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_stream():
    global is_streaming
    is_streaming = True
    return jsonify({"status": "started"})

@app.route('/stop', methods=['POST'])
def stop_stream():
    global is_streaming
    is_streaming = False
    return jsonify({"status": "stopped"})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/confidence_data')
def confidence_data():
    return jsonify(confidence_scores)

if __name__ == '__main__':
    app.run(debug=True)

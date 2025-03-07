import os
import json
import numpy as np
import tensorflow as tf
import mediapipe as mp  # Add MediaPipe import
from flask import Flask, render_template, jsonify, request
import base64
import cv2
import gc

# Set up MediaPipe components
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Holistic model
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# ตั้งค่า TensorFlow ให้ใช้หน่วยความจำเท่าที่จำเป็น
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# โหลดโมเดลอย่างมีประสิทธิภาพ
model = tf.keras.models.load_model("flip_hand_gesture_model.h5", compile=False)

# Load gesture classes
with open("gesture_classes.json", 'r') as f:
    gesture_names = json.load(f)

# Thai translations for gesture names
thai_translations = {
    "Dangerous": "อันตราย",
    "Dormitory": "หอพัก",
    "Hello": "สวัสดี",
    "Hospital": "โรงพยาบาล",
    "I": "ฉัน",
    "Love": "รัก",
    "Nickname": "ชื่อเล่น",
    "Overpass": "สะพานลอย",
    "Police Station": "สถานีตำรวจ",
    "Sad": "เศร้า",
    "Sleepy": "ง่วงนอน",
    "Sorry": "ขอโทษ",
    "Speak": "พูด",
    "Stressed": "เครียด",
    "Thank You": "ขอบคุณ",
    "University": "มหาวิทยาลัย",
    "Unwell": "ไม่สบาย",
    "What time": "กี่โมง",
    "Worried": "กังวล",
    "You": "คุณ",
    "Unknown": "ไม่รู้จัก"
}

# Function to extract landmarks
def extract_landmarks(frame):
    # Define upper body indices
    upper_body_indices = [
        0,  # nose
        11, 12,  # shoulders
        13, 14, 15, 16,  # arms
        23, 24,  # hips
        19, 20,  # elbows
        21, 22,  # wrists
    ]
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)
    landmarks = []
    
    # 1. Extract Hand landmarks (21 points per hand)
    if results.left_hand_landmarks:
        for landmark in results.left_hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
    else:
        landmarks.extend([0] * (21 * 3))  # Pad with zeros if no left hand detected
        
    if results.right_hand_landmarks:
        for landmark in results.right_hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
    else:
        landmarks.extend([0] * (21 * 3))  # Pad with zeros if no right hand detected
    
    # 2. Extract Face mesh landmarks (468 points)
    if results.face_landmarks:
        for landmark in results.face_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
    else:
        landmarks.extend([0] * (468 * 3))  # Pad with zeros if no face detected
    
    # 3. Extract Upper body landmarks
    if results.pose_landmarks:
        # Extract upper body points (shoulders, arms, torso)
        for idx in upper_body_indices:
            landmark = results.pose_landmarks.landmark[idx]
            landmarks.extend([landmark.x, landmark.y, landmark.z])
    else:
        landmarks.extend([0] * (len(upper_body_indices) * 3))
    
    return landmarks

# Function to get gesture name
def get_gesture_name(class_idx):
    english_name = gesture_names[class_idx] if 0 <= class_idx < len(gesture_names) else "Unknown"
    # Return Thai translation if available, otherwise return English name
    return thai_translations.get(english_name, english_name)

# Create Flask app
app = Flask(__name__)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/examples')
def examples():
    return render_template('examples.html')

@app.route('/gesture')
def gesture_example():
    """Render the page for a single gesture example with video."""
    return render_template('gesture.html')

@app.route('/process_frames', methods=['POST'])
def process_frames():
    try:
        # รับข้อมูล
        data = request.json
        frames_data = data.get('frames', [])
        
        if not frames_data:
            return jsonify({"status": "error", "message": "ไม่พบข้อมูลเฟรม"})
        
        # Modified to process frames correctly
        # Check if frames_data contains raw images (from frontend) or already extracted landmarks
        if isinstance(frames_data[0], list) and all(isinstance(x, (int, float)) for x in frames_data[0]):
            # These are already landmarks
            sequence = np.array([frames_data], dtype=np.float32)
        else:
            # These are base64 encoded images, need to process them
            processed_frames = []
            for frame_b64 in frames_data:
                # Decode base64 to image
                frame_data = base64.b64decode(frame_b64.split(',')[1])
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Extract landmarks using MediaPipe
                landmarks = extract_landmarks(frame)
                processed_frames.append(landmarks)
            
            sequence = np.array([processed_frames], dtype=np.float32)
        
        # ทำนายผลด้วยการตั้งค่าที่เหมาะสม
        prediction = model.predict(sequence, verbose=0, batch_size=1)
        class_idx = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        # ล้างหน่วยความจำ
        del sequence
        gc.collect()
        
        # ได้ผลลัพธ์
        PREDICTION_THRESHOLD = 0.8
        if confidence > PREDICTION_THRESHOLD:
            predicted_gesture = get_gesture_name(class_idx)
        else:
            predicted_gesture = "ไม่แน่ใจ"
        
        return jsonify({
            "status": "success", 
            "prediction": predicted_gesture,
            "confidence": float(confidence)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    # Get port from environment variable for Render
    port = int(os.environ.get('PORT', 5000))
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=False)

import cv2
import numpy as np
import os
import json
import tensorflow as tf
import mediapipe as mp
from flask import Flask, render_template, Response, jsonify, request

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

# Load the trained model
model = tf.keras.models.load_model("flip_hand_gesture_model.h5")

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize holistic model
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Global variables for configuration
show_landmarks = False
show_text = False

def extract_landmarks(frame):
    # Define upper body indices outside the conditional block
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
    
    return landmarks, results

def draw_landmarks(frame, results):
    global show_landmarks
    
    # Only draw landmarks if show_landmarks is True
    if not show_landmarks:
        return
    
    # Draw hand landmarks
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    
    # Draw face mesh
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
    
    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing_styles.get_default_pose_landmarks_style())

def get_gesture_name(class_idx):
    english_name = gesture_names[class_idx] if 0 <= class_idx < len(gesture_names) else "Unknown"
    # Return Thai translation if available, otherwise return English name
    return thai_translations.get(english_name, english_name)

# Create Flask app
app = Flask(__name__)

# Global variables for gesture recognition
sequence = []
is_recording = False
recording_frames = 0
SEQUENCE_LENGTH = 30
PREDICTION_THRESHOLD = 0.8
final_prediction = None
waiting_for_hand = False

# Create a generator for the video feed
def generate_frames():
    global sequence, is_recording, recording_frames, final_prediction, waiting_for_hand, show_text
    
    cap = cv2.VideoCapture(0)  # Use 0 for default camera, change if needed
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)  # Mirror the image
        display_frame = frame.copy()
        
        # Process frame with MediaPipe
        landmarks, results = extract_landmarks(frame)
        
        # Draw the landmarks on the frame (only if show_landmarks is True)
        draw_landmarks(display_frame, results)
        
        # Check for hand detection when waiting
        if waiting_for_hand:
            if results.left_hand_landmarks or results.right_hand_landmarks:
                is_recording = True
                waiting_for_hand = False
            else:
                # Only show text if show_text is True
                if show_text:
                    cv2.putText(display_frame, "ไม่พบมือ กรุณาแสดงมือให้กล้องเห็น",
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Only record frames if we're recording and have detected hands
        if is_recording and recording_frames < SEQUENCE_LENGTH:
            if results.left_hand_landmarks or results.right_hand_landmarks:
                sequence.append(landmarks)
                recording_frames += 1
                
                # Only show text if show_text is True
                if show_text:
                    cv2.putText(display_frame, f"เริ่มบันทึก: {recording_frames}/{SEQUENCE_LENGTH}",
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # If we lose hand detection during recording
                if show_text:
                    cv2.putText(display_frame, "ไม่พบมือ กรุณาแสดงมือให้กล้องเห็น",
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if recording_frames == SEQUENCE_LENGTH:
                is_recording = False
                sequence_array = np.array([sequence])
                prediction = model.predict(sequence_array, verbose=0)
                class_idx = np.argmax(prediction[0])
                confidence = np.max(prediction[0])
                
                if confidence > PREDICTION_THRESHOLD:
                    final_prediction = get_gesture_name(class_idx)
                else:
                    final_prediction = "ไม่แน่ใจ"
        
        # Draw the prediction on the frame (only if show_text is True)
        if final_prediction and show_text:
            cv2.putText(display_frame, f"คำที่ทำนายได้: {final_prediction}",
                      (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame_bytes = buffer.tobytes()
        
        # Yield the frame for the response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

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

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recording')
def start_recording():
    global sequence, is_recording, recording_frames, final_prediction, waiting_for_hand
    sequence = []
    recording_frames = 0
    final_prediction = None
    is_recording = False
    waiting_for_hand = True
    return jsonify({"status": "waiting_for_hand"})

@app.route('/clear')
def clear():
    global sequence, is_recording, recording_frames, final_prediction, waiting_for_hand
    sequence = []
    is_recording = False
    waiting_for_hand = False
    recording_frames = 0
    final_prediction = None
    return jsonify({"status": "cleared"})

@app.route('/get_prediction')
def get_prediction():
    global final_prediction, recording_frames
    return jsonify({
        "prediction": final_prediction if final_prediction else "None",
        "frames": recording_frames
    })

@app.route('/toggle_landmarks', methods=['POST'])
def toggle_landmarks():
    global show_landmarks
    data = request.get_json()
    show_landmarks = data.get('show', show_landmarks)
    return jsonify({"status": "success", "show_landmarks": show_landmarks})

@app.route('/toggle_text', methods=['POST'])
def toggle_text():
    global show_text
    data = request.get_json()
    show_text = not data.get('hide', False)  # Invert the hide value
    return jsonify({"status": "success", "show_text": show_text})

if __name__ == '__main__':
    # กำหนดพอร์ตจาก environment variable สำหรับ Render
    port = int(os.environ.get('PORT', 5000))
    # แก้ไขพารามิเตอร์ในการรัน app
    app.run(host='0.0.0.0', port=port, debug=False)
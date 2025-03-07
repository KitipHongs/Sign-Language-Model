import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, jsonify, request
import base64
import cv2
import gc

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

# Load the trained model
model = tf.keras.models.load_model("flip_hand_gesture_model.h5")

# Create Flask app
app = Flask(__name__)

# Function to get gesture name
def get_gesture_name(class_idx):
    english_name = gesture_names[class_idx] if 0 <= class_idx < len(gesture_names) else "Unknown"
    # Return Thai translation if available, otherwise return English name
    return thai_translations.get(english_name, english_name)

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
        
        # แปลงเป็น numpy array อย่างมีประสิทธิภาพและตรวจสอบรูปร่างข้อมูล
        sequence = np.array([frames_data], dtype=np.float32)  # ระบุ dtype เพื่อประหยัดหน่วยความจำ
        
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

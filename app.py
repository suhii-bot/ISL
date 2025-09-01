#!/usr/bin/env python3
"""
ISL (Indian Sign Language) Translator Web Application
A Flask-based web app for real-time sign language recognition and dictionary display.

Features:
- Live webcam prediction using trained CNN model
- Dictionary page with A-Z and 0-9 sign cards
- Dark/Light theme toggle
- Image upload functionality for custom sign images

Author: ISL Project Team
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, request, jsonify, Response, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import base64
import json
from threading import Lock
import time

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'isl-translator-secret-key-2024'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/dictionary_images', exist_ok=True)

# Global variables for model and MediaPipe
model = None
label_encoder = None
mp_hands = None
hands = None
mp_draw = None
model_lock = Lock()

# Allowed file extensions for image upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_isl_model():
    """Load the trained ISL model and setup MediaPipe"""
    global model, label_encoder, mp_hands, hands, mp_draw
    
    try:
        # Load the trained model
        model_path = "model/cnn_model.h5"
        if os.path.exists(model_path):
            model = load_model(model_path)
            print("✅ Model loaded successfully")
        else:
            print("❌ Model file not found. Please train the model first.")
            return False
        
        # Load the dataset for label encoding
        data_path = "data/isl_combined_data.csv"
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            labels = df['label'].astype(str).unique()
            label_encoder = LabelEncoder()
            label_encoder.fit(labels)
            print(f"✅ Label encoder setup with {len(labels)} classes: {sorted(labels)}")
        else:
            print("❌ Dataset file not found")
            return False
        
        # Setup MediaPipe Hands
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        mp_draw = mp.solutions.drawing_utils
        print("✅ MediaPipe setup complete")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

def extract_landmarks(frame):
    """Extract hand landmarks from frame"""
    global hands
    
    if hands is None:
        return None, frame
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    landmark_list = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract landmark coordinates
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])
    
    # Pad with zeros if less than 2 hands detected (126 features total)
    while len(landmark_list) < 126:
        landmark_list.extend([0.0, 0.0, 0.0])
    
    return landmark_list[:126], frame

def predict_sign(landmarks):
    """Predict sign from landmarks using the trained model"""
    global model, label_encoder
    
    if model is None or label_encoder is None:
        return "Model not loaded", 0.0
    
    try:
        # Make prediction
        prediction = model.predict(np.array([landmarks]), verbose=0)[0]
        predicted_index = np.argmax(prediction)
        confidence = prediction[predicted_index]
        
        # Get label
        try:
            predicted_label = label_encoder.inverse_transform([predicted_index])[0]
        except ValueError:
            predicted_label = "Unknown"
        
        return predicted_label, float(confidence)
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return "Error", 0.0

@app.route('/')
def index():
    """Homepage with live webcam prediction"""
    return render_template('index.html')

@app.route('/dictionary')
def dictionary():
    """Dictionary page showing A-Z and 0-9 signs"""
    # Define all available signs
    alphabets = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    numbers = [str(i) for i in range(10)]
    all_signs = alphabets + numbers
    
    return render_template('dictionary.html', signs=all_signs)

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    """API endpoint to predict sign from webcam frame"""
    try:
        # Get image data from request
        data = request.get_json()
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64, prefix
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Extract landmarks
        landmarks, processed_frame = extract_landmarks(frame)
        
        if landmarks and len(landmarks) == 126:
            # Predict sign
            predicted_label, confidence = predict_sign(landmarks)
            
            return jsonify({
                'success': True,
                'prediction': predicted_label,
                'confidence': round(confidence * 100, 1)
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No hands detected or invalid landmarks'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/upload_sign_image', methods=['POST'])
def upload_sign_image():
    """Handle custom image upload for signs"""
    try:
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
            
        file = request.files['image']
        sign = request.form.get('sign', '').upper()
        
        if not sign:
            return jsonify({
                'success': False,
                'error': 'No sign specified'
            }), 400
            
        if file and allowed_file(file.filename):
            # Create directory if it doesn't exist
            os.makedirs('static/dictionary_images', exist_ok=True)
            
            # Save file with sign name
            ext = file.filename.rsplit('.', 1)[1].lower()
            filename = f"{sign}.{ext}"
            filepath = os.path.join('static/dictionary_images', filename)
            
            # Remove existing images for this sign
            for existing_ext in ALLOWED_EXTENSIONS:
                existing_file = os.path.join('static/dictionary_images', f"{sign}.{existing_ext}")
                if os.path.exists(existing_file):
                    os.remove(existing_file)
            
            # Save new image
            file.save(filepath)
            
            return jsonify({
                'success': True,
                'message': f'Image uploaded successfully for sign {sign}',
                'image_url': f'/static/dictionary_images/{filename}'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid file type'
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/toggle_theme', methods=['POST'])
def toggle_theme():
    """Toggle between light and dark theme"""
    try:
        data = request.get_json()
        theme = data.get('theme', 'light')
        
        return jsonify({
            'success': True,
            'theme': theme
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/get_sign_image/<sign>')
def get_sign_image(sign):
    """Get custom image for a specific sign"""
    sign = sign.upper()
    
    # Check for different file extensions
    for ext in ['png', 'jpg', 'jpeg', 'gif', 'webp']:
        filename = f"{sign}.{ext}"
        filepath = os.path.join('static/dictionary_images', filename)
        
        if os.path.exists(filepath):
            return jsonify({
                'success': True,
                'image_url': f'/static/dictionary_images/{filename}'
            })
    
    return jsonify({'success': False, 'error': 'No custom image found'})

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('500.html'), 500

if __name__ == '__main__':
    print("🚀 Starting ISL Translator Web Application...")
    
    # Load the model and setup MediaPipe
    if load_isl_model():
        print("✅ Application ready!")
        print("📱 Open your browser and go to: http://localhost:5000")
        print("🎯 Features:")
        print("   - Homepage: Live webcam sign prediction")
        print("   - Dictionary: A-Z and 0-9 sign cards with image upload")
        print("   - Theme toggle: Dark/Light mode")
        print("\n🛑 Press Ctrl+C to stop the server")
        
        # Run the Flask app
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    else:
        print("❌ Failed to load model. Please check the model file and try again.")
        print("💡 Make sure you have trained the model first using train_cnn.py")
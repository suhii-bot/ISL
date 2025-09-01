import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import time as time_module
from collections import deque


# Load gesture recognition model
model = load_model("model/cnn_model.h5")


# Load labels
df = pd.read_csv("data/isl_combined_data.csv")
labels = df['label'].astype(str).unique()
le = LabelEncoder()
le.fit(labels)


# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils


# Camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)


# State variables
last_label = None
last_spoken_time = 0
confidence_threshold = 0.85
prediction_buffer = deque(maxlen=5)


print("🟢 Real-time ISL prediction started. Press ESC to exit.")


while True:
    ret, frame = cap.read()
    if not ret:
        continue


    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)


    landmark_list = []


    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])


        while len(landmark_list) < 126:
            landmark_list.extend([0.0, 0.0, 0.0])


        if len(landmark_list) == 126:
            prediction = model.predict(np.array([landmark_list]), verbose=0)[0]
            predicted_index = np.argmax(prediction)
            confidence = prediction[predicted_index]


            try:
                predicted_label = le.inverse_transform([predicted_index])[0]
            except ValueError:
                predicted_label = "Unknown"


            prediction_buffer.append(predicted_label)
            smooth_label = max(set(prediction_buffer), key=prediction_buffer.count)


            cv2.putText(frame, f"{smooth_label} ({confidence*100:.1f}%)",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)


            current_time = time_module.time()


            print(f"Detected: {smooth_label}, Last spoken: {last_label}, "
                  f"Conf: {confidence:.2f}, Time since last: {current_time - last_spoken_time:.2f}s")


            # Speak only when gesture changes, never repeats for the same gesture
            if confidence > confidence_threshold:
                if smooth_label != last_label:
                    speak_engine = pyttsx3.init()
                    speak_engine.setProperty('rate', 150)
                    print(f"🔊 Speaking (changed gesture): {smooth_label}")
                    speak_engine.say(smooth_label)
                    speak_engine.runAndWait()
                    del speak_engine
                    last_spoken_time = current_time
                    last_label = smooth_label


    cv2.imshow("ISL Real-Time Prediction with Voice", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()

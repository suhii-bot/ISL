import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load your trained model
model = load_model("model/cnn_model.h5")

# Load the exact dataset used for training (merged numbers + alphabets)
df = pd.read_csv("data/isl_combined_data.csv")  # change path/filename as per your merged data

# Ensure labels are all strings to avoid type mismatch
labels = df['label'].astype(str).unique()
le = LabelEncoder()
le.fit(labels)

# Setup MediaPipe Hands (max 2 hands)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)  # mirror image for natural feel
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    landmark_list = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

        # Pad zeros if only 1 hand detected (needed shape: 126)
        while len(landmark_list) < 126:
            landmark_list.extend([0.0, 0.0, 0.0])

        if len(landmark_list) == 126:
            prediction = model.predict(np.array([landmark_list]), verbose=0)[0]
            predicted_index = np.argmax(prediction)

            # Safe inverse transform
            try:
                predicted_label = le.inverse_transform([predicted_index])[0]
            except ValueError:
                predicted_label = "Unknown"

            confidence = prediction[predicted_index]

            cv2.putText(
                frame,
                f"{predicted_label} ({confidence*100:.1f}%)",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                2
            )

    cv2.imshow("ISL Prediction", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()

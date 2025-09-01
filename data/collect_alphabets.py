import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# MediaPipe Hands setup (two hands)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,          # Changed to 2 hands
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

data = []
labels = []
sample_count = 0
collecting = False
current_label = ''

print("📸 Press a key (A-Z) to start collecting 50 samples of that alphabet.")
print("Press ENTER to save data, ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        all_landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for lm in hand_landmarks.landmark:
                all_landmarks.extend([lm.x, lm.y, lm.z])

        # Pad zeros if only one hand detected (2 hands x 21 landmarks x 3 = 126 values)
        while len(all_landmarks) < 126:
            all_landmarks.extend([0.0, 0.0, 0.0])

        if collecting and len(all_landmarks) == 126:
            data.append(all_landmarks)
            labels.append(current_label)
            sample_count += 1
            print(f"✅ Sample {sample_count}/50 for '{current_label}' captured")

            if sample_count >= 50:
                collecting = False
                sample_count = 0
                print(f"🎉 Done collecting 50 samples for: '{current_label}'")

    cv2.putText(frame, f"Label: {current_label} | Samples: {sample_count}/50", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("ISL Alphabet Collector (2 Hands)", frame)
    key = cv2.waitKey(10)

    if key != -1:
        if key == 27:  # ESC to exit
            break
        elif key == 13:  # ENTER to save data
            if data:
                df = pd.DataFrame(data)
                df['label'] = labels
                os.makedirs("data", exist_ok=True)
                df.to_csv("data/isl_alphabets_data.csv", index=False)
                print("💾 Alphabet data saved to 'data/isl_alphabets_data.csv'")
                data.clear()
                labels.clear()
        else:
            char = chr(key).upper()
            if char.isalpha():
                current_label = char
                collecting = True
                sample_count = 0
                print(f"📌 Started collecting for: '{current_label}'")

cap.release()
cv2.destroyAllWindows()

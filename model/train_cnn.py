import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# ---------------------------
# 1. Load cleaned dataset
# ---------------------------
df = pd.read_csv("data/isl_combined_data.csv")

X = df.drop('label', axis=1).values  # landmark features
y = df['label'].astype(str)          # gesture labels

# ---------------------------
# 2. Encode labels
# ---------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save label classes for later use in prediction
np.save("model/label_classes.npy", le.classes_)

# Convert labels to one-hot form
y_categorical = to_categorical(y_encoded)

# ---------------------------
# 3. Train/test split
# ---------------------------
X_train, X_test, y_train, y_test, y_train_labels, y_test_labels = train_test_split(
    X, y_categorical, y_encoded,
    test_size=0.15,
    random_state=42,
    stratify=y_encoded
)

# ---------------------------
# 4. Model architecture
# ---------------------------
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.2),

    Dense(len(le.classes_), activation='softmax')
])

optimizer = Adam(learning_rate=0.0008)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# ---------------------------
# 5. Callbacks
# ---------------------------
checkpoint = ModelCheckpoint("model/cnn_model.h5", save_best_only=True,
                              monitor='val_accuracy', mode='max', verbose=1)
early_stop = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)

# ---------------------------
# 6. Train the model
# ---------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=32,
    epochs=150,
    callbacks=[checkpoint, early_stop],
    verbose=1
)

# ---------------------------
# 7. Final evaluation
# ---------------------------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Final Test Accuracy: {acc*100:.2f}%")
print("📁 Model saved to: model/cnn_model.h5\n")

# ---------------------------
# 8. Confusion Matrix & Classification Report
# ---------------------------
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test_labels, y_pred)
report = classification_report(y_test_labels, y_pred, target_names=le.classes_, digits=3)
acc_report = accuracy_score(y_test_labels, y_pred)

print("=== Classification Report (by gesture) ===")
print(report)
print(f"Overall Test Accuracy: {acc_report*100:.2f}%")

# Save & show confusion matrix plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=le.classes_,
           yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True Label')
plt.title('Gesture Confusion Matrix')
plt.tight_layout()
plt.savefig("model/confusion_matrix.png")
plt.show()

print("🖼️ Confusion matrix saved to model/confusion_matrix.png")

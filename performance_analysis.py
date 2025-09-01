import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split, learning_curve

# 1. Load dataset & model
data_path = "data/isl_combined_data.csv"
model_path = "model/cnn_model.h5"

df = pd.read_csv(data_path)

X = df.drop('label', axis=1).values
y = df['label'].astype(str)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

X_train, X_test, y_train, y_test, y_train_labels, y_test_labels = train_test_split(
    X, y_categorical, y_encoded,
    test_size=0.15,
    random_state=42,
    stratify=y_encoded
)

model = load_model(model_path)
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Overall Test Accuracy: {acc*100:.2f}%\n")

y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

report = classification_report(
    y_test_labels,
    y_pred,
    target_names=le.classes_,
    digits=3
)
print("=== Classification Report ===")
print(report)

cm = confusion_matrix(y_test_labels, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("model/confusion_matrix.png")
plt.show()

print("\n🖼️ Confusion Matrix saved to: model/confusion_matrix.png")

# 6. Regression Visualization (regression line)
binary_labels = (y_test_labels == 0).astype(int)
probs_class0 = y_pred_probs[:, 0]

reg = LinearRegression()
reg.fit(probs_class0.reshape(-1, 1), binary_labels)
y_reg_pred = reg.predict(probs_class0.reshape(-1, 1))

plt.scatter(probs_class0, binary_labels, alpha=0.4)
plt.plot(probs_class0, y_reg_pred, color='red')
plt.xlabel("Predicted Probability (Class 0)")
plt.ylabel("True Label (Class 0)")
plt.title("Regression Line: Predicted Probability vs True Label")
plt.tight_layout()
plt.savefig("model/regression_line.png")
plt.show()

# 7. KNN Decision Boundary Visualization (PCA reduction)
pca = PCA(n_components=2)
X_test_2d = pca.fit_transform(X_test)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_test_2d, y_test_labels)

h = .05
x_min, x_max = X_test_2d[:, 0].min() - 1, X_test_2d[:, 0].max() + 1
y_min, y_max = X_test_2d[:, 1].min() - 1, X_test_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
mesh_points = np.c_[xx.ravel(), yy.ravel()]
Z = knn.predict(mesh_points)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='rainbow')
plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test_labels, cmap='rainbow', edgecolor='k')
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("KNN Decision Boundary (2D PCA)")
plt.tight_layout()
plt.savefig("model/knn_boundary.png")
plt.show()

# 8. ROC Curve & AUC (multiclass)
n_classes = y_pred_probs.shape[1]
y_test_bin = label_binarize(y_test_labels, classes=np.arange(n_classes))

plt.figure(figsize=(10, 8))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
    plt.plot(fpr, tpr, label='Class %s (AUC %.2f)' % (le.classes_[i], auc(fpr, tpr)))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig("model/roc_curve.png")
plt.show()

# 9. Precision-Recall Curve (multiclass)
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_pred_probs[:, i])
    plt.plot(recall, precision, label='Class %s' % le.classes_[i])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.tight_layout()
plt.savefig("model/precision_recall.png")
plt.show()

# 10. Learning Curve
from sklearn.model_selection import ShuffleSplit
from sklearn.dummy import DummyClassifier

train_sizes = np.linspace(0.1, 1.0, 10)
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
dummy = DummyClassifier(strategy='most_frequent')
X_flat, y_flat = X, y_encoded
train_sizes, train_scores, test_scores = learning_curve(
    dummy, X_flat, y_flat, train_sizes=train_sizes, cv=cv, scoring='accuracy'
)
plt.figure(figsize=(8,6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='r', label='Training accuracy')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color='g', label='Validation accuracy')
plt.title("Learning Curve (Dummy Reference)")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("model/learning_curve.png")
plt.show()

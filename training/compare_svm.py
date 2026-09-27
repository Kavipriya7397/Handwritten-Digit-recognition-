import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ==========================================
# 1. LOAD MNIST DATASET
# ==========================================

print("Loading MNIST dataset...")

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Training samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])


# ==========================================
# 2. PREPROCESS FOR SVM
# ==========================================

# Flatten 28x28 images into 784 features
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Normalize pixel values
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0


# ==========================================
# 3. TRAIN SVM BASELINE
# ==========================================

print("\nTraining SVM baseline...")

svm_model = LinearSVC(
    C=1.0,
    max_iter=3000,
    random_state=42
)

svm_model.fit(X_train, y_train)


# ==========================================
# 4. PREDICTION
# ==========================================

print("\nGenerating SVM predictions...")

y_pred = svm_model.predict(X_test)


# ==========================================
# 5. PERFORMANCE METRICS
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)


print("\n========== SVM PERFORMANCE ==========")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")

print("====================================")


# ==========================================
# 6. CNN RESULT
# ==========================================

project_root = Path(__file__).resolve().parents[1]
cnn_model = load_model(
    project_root / "model" / "digit_cnn.keras",
    compile=False,
)
cnn_predictions = cnn_model.predict(
    X_test.reshape(-1, 28, 28, 1),
    verbose=0,
)
cnn_pred = np.argmax(cnn_predictions, axis=1)
cnn_accuracy = accuracy_score(y_test, cnn_pred) * 100
cnn_precision = precision_score(y_test, cnn_pred, average="weighted") * 100
cnn_recall = recall_score(y_test, cnn_pred, average="weighted") * 100
cnn_f1 = f1_score(y_test, cnn_pred, average="weighted") * 100


# ==========================================
# 7. COMPARISON GRAPH
# ==========================================

results_dir = project_root / "results"
results_dir.mkdir(exist_ok=True)

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score"
]

cnn_values = [
    cnn_accuracy,
    cnn_precision,
    cnn_recall,
    cnn_f1
]

svm_values = [
    accuracy * 100,
    precision * 100,
    recall * 100,
    f1 * 100
]


x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(9, 6))

plt.bar(
    x - width / 2,
    cnn_values,
    width,
    label="CNN"
)

plt.bar(
    x + width / 2,
    svm_values,
    width,
    label="SVM"
)

plt.xticks(x, metrics)
plt.ylabel("Score (%)")
plt.ylim(80, 100)
plt.title("CNN vs SVM Performance Comparison")
plt.legend()

plt.tight_layout()

plt.savefig(
    results_dir / "cnn_vs_svm.png",
    dpi=300
)

plt.close()

with (results_dir / "comparison_metrics.json").open("w", encoding="utf-8") as metrics_file:
    json.dump(
        {
            "cnn": {
                "Accuracy": cnn_accuracy,
                "Precision": cnn_precision,
                "Recall": cnn_recall,
                "F1-Score": cnn_f1,
            },
            "svm": {
                "Accuracy": accuracy * 100,
                "Precision": precision * 100,
                "Recall": recall * 100,
                "F1-Score": f1 * 100,
            },
        },
        metrics_file,
        indent=2,
    )


print("\nComparison graph saved!")
print("Location: results/cnn_vs_svm.png")

print("\nSVM baseline completed successfully!")
import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR = PROJECT_ROOT / "model"


# ==========================================
# 1. LOAD MNIST DATASET
# ==========================================

print("Loading MNIST dataset...")

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Training images:", X_train.shape)
print("Testing images :", X_test.shape)


# ==========================================
# 2. PREPROCESSING
# ==========================================

# Normalize pixel values from 0-255 to 0-1
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Add channel dimension
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Convert labels to one-hot encoding
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)


# ==========================================
# 3. BUILD CNN MODEL
# ==========================================

print("\nBuilding CNN model...")

model = Sequential([
    
    # First convolution block
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(28, 28, 1)
    ),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Second convolution block
    Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Third convolution block
    Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),
    BatchNormalization(),

    # Convert feature maps into vector
    Flatten(),

    # Fully connected layer
    Dense(128, activation="relu"),

    # Prevent overfitting
    Dropout(0.5),

    # Output layer: 10 digits
    Dense(10, activation="softmax")
])


# ==========================================
# 4. COMPILE MODEL
# ==========================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ==========================================
# 5. CALLBACKS
# ==========================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.00001
)


# ==========================================
# 6. TRAIN MODEL
# ==========================================

print("\nTraining CNN...")

history = model.fit(
    X_train,
    y_train_cat,
    validation_split=0.1,
    epochs=15,
    batch_size=128,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)


# ==========================================
# 7. EVALUATE MODEL
# ==========================================

print("\nEvaluating model...")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test_cat,
    verbose=0
)

print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss    : {test_loss:.4f}")


# ==========================================
# 8. PREDICTIONS
# ==========================================

y_probability = model.predict(X_test, verbose=0)

y_pred = np.argmax(y_probability, axis=1)


# ==========================================
# 9. PERFORMANCE METRICS
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

RESULTS_DIR.mkdir(exist_ok=True)
with (RESULTS_DIR / "metrics.json").open("w", encoding="utf-8") as metrics_file:
    json.dump(
        {
            "test_accuracy": float(test_accuracy),
            "test_loss": float(test_loss),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        },
        metrics_file,
        indent=2,
    )

print("\n========== PERFORMANCE METRICS ==========")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")
print("==========================================")


# ==========================================
# 10. CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==========================================
# 11. SAVE ACCURACY / LOSS GRAPH
# ==========================================

RESULTS_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "accuracy_graph.png",
    dpi=300
)

plt.close()


# ==========================================
# 12. SAVE LOSS GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("CNN Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "loss_graph.png",
    dpi=300
)

plt.close()


# ==========================================
# 13. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("CNN Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=300
)

plt.close()


# ==========================================
# 14. SAVE MODEL
# ==========================================

model.save(
    MODEL_DIR / "digit_cnn.keras"
)

print("\nModel saved successfully!")
print("Location: model/digit_cnn.keras")

print("\nTraining completed successfully!")
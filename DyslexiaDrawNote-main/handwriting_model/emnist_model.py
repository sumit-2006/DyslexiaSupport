import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

# Adjust the path if needed
MODEL_PATH = "emnist_model.h5"  
LABEL_MAP  = {i: chr(65+i) for i in range(26)}    # Example: 0→A, 1→B… adjust to your 47 classes

_model = None
def load_emnist():
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH)
    return _model


def predict_emnist(char_img):
    model = load_emnist()

    # Resize to 28x28
    resized = cv2.resize(char_img, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalize and expand dimensions
    x = resized.astype('float32') / 255.0
    x = np.expand_dims(x, axis=-1)  # Add channel dim -> (28, 28, 1)
    x = np.expand_dims(x, axis=0)   # Add batch dim -> (1, 28, 28, 1)

    char_logits, flip_logits = model.predict(x, verbose=0)
    label_idx = np.argmax(char_logits)
    confidence = float(np.max(char_logits))
    label = chr(label_idx + ord('a'))  # Or however your label mapping works

    return label, confidence


import numpy as np
from tensorflow.keras.models import load_model
import cv2

MODEL_PATH = "reversal_detector.h5"
_reversal_model = None

def load_reversal():
    global _reversal_model
    if _reversal_model is None:
        _reversal_model = load_model(MODEL_PATH)
    return _reversal_model

def predict_reversal(img: np.ndarray):
    model = load_reversal()

    # Resize to 64x64
    h, w = img.shape
    scale = 64 / max(h, w)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    pad_h = 64 - resized.shape[0]
    pad_w = 64 - resized.shape[1]
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    padded = padded.astype(np.float32) / 255.0
    x = np.expand_dims(padded, axis=-1)  # (64, 64, 1)
    x = np.expand_dims(x, axis=0)        # (1, 64, 64, 1)

    assert x.shape == (1, 64, 64, 1), f"Expected input shape (1, 64, 64, 1), got {x.shape}"

    prob = model.predict(x, verbose=0)[0, 0]
    return float(prob)

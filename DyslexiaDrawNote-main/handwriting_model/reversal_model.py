import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "handwriting_model/reversal_detector.h5"
_reversal_model = None

def load_reversal():
    global _reversal_model
    if _reversal_model is None:
        _reversal_model = load_model(MODEL_PATH)
    return _reversal_model

def predict_reversal(img: np.ndarray):
    """
    img: single-channel 32×32 np.uint8 crop
    returns: flip_prob:float
    """
    model = load_reversal()
    x = img.astype(np.float32)/255.0
    x = np.expand_dims(x, (0, -1))   # shape (1,32,32,1)
    prob = model.predict(x)[0,0]
    return float(prob)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Adjust the path if needed
MODEL_PATH = "handwriting_model/emnist_dualhead.h5"  
LABEL_MAP  = {i: chr(65+i) for i in range(26)}    # Example: 0→A, 1→B… adjust to your 47 classes

_model = None
def load_emnist():
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH)
    return _model

def predict_emnist(img: np.ndarray):
    """
    img: single-channel 28×28 np.uint8 crop
    returns: (char_label:str, flip_prob:float)
    """
    model = load_emnist()
    x = img.astype(np.float32)/255.0
    x = np.expand_dims(x, (0, -1))   # shape (1,28,28,1)
    char_logits, flip_logits = model.predict(x)
    char_idx = np.argmax(char_logits, axis=1)[0]
    flip_prob = float(tf.sigmoid(flip_logits)[0,0])
    return LABEL_MAP[char_idx], flip_prob

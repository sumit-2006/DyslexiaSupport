from handwriting_model.emnist_model import predict_emnist
from handwriting_model.reversal_model import predict_reversal

import cv2

def evaluate_char(img_path: str):
    """
    Loads the image, runs both models, and returns:
      {
        'char': 'A',
        'flip_emnist': 0.12,
        'flip_binary': 0.08
      }
    """
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
      raise ValueError(f"Failed to load image: {img_path}")


    # resize/preprocess to each model’s input
    emnist_in  = cv2.resize(gray, (28,28), interpolation=cv2.INTER_AREA)
    binary_in  = cv2.resize(gray, (32,32), interpolation=cv2.INTER_AREA)

    char, p_emnist = predict_emnist(emnist_in)
    p_binary     = predict_reversal(binary_in)

    return {
        'char': char,
        'flip_emnist': p_emnist,
        'flip_binary': p_binary
    }

if __name__ == "__main__":
    import os, json
    test_dir = "server/uploads/chars/word_0"
    results = {}
    for fn in os.listdir(test_dir):
        path = os.path.join(test_dir, fn)
        results[fn] = evaluate_char(path)
    print(json.dumps(results, indent=2))

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import base64
# import io
# from PIL import Image
# import numpy as np
# import tensorflow as tf

# app = Flask(__name__)
# CORS(app)  # Correct CORS setup

# # Load your model
# model = tf.keras.models.load_model('flipped_emnist_classifier.h5')

# # Preprocessing function
# def preprocess_image(base64_image, target_size=(28, 28)):
#     image_data = base64.b64decode(base64_image.split(',')[1])
#     image = Image.open(io.BytesIO(image_data)).convert('L')
#     image = image.resize(target_size)
#     image = np.array(image).astype('float32') / 255.0
#     image = np.expand_dims(image, axis=0)
#     image = np.expand_dims(image, axis=-1)
#     return image

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     image_base64 = data.get('image')

#     if not image_base64:
#         return jsonify({'error': 'No image data received'}), 400

#     try:
#         preprocessed = preprocess_image(image_base64)
#         prediction = model.predict(preprocessed)
#         predicted_class = int(np.argmax(prediction))
#         return jsonify({'prediction': predicted_class})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True, port=8000)

from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
import numpy as np
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load your trained model
model = tf.keras.models.load_model('flipped_emnist_classifier.h5')

# Label mapping (62 EMNIST classes)
label_to_char = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
    36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j',
    46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o', 51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't',
    56: 'u', 57: 'v', 58: 'w', 59: 'x', 60: 'y', 61: 'z'
}
num_classes = 62  # EMNIST base classes

def preprocess_base64_image(base64_image, target_size=(28, 28)):
    try:
        image_data = base64.b64decode(base64_image.split(',')[1])
        image = Image.open(io.BytesIO(image_data)).convert('L')
        image = image.resize(target_size)
        image = np.array(image).astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)  # shape (1, 28, 28)
        image = np.expand_dims(image, axis=-1) # shape (1, 28, 28, 1)
        return image
    except Exception as e:
        raise ValueError(f"Failed to process image: {e}")

def predict_and_correct(image_tensor):
    preds = model.predict(image_tensor)
    pred_label = int(np.argmax(preds))

    flipped = False
    if pred_label >= num_classes:
        flipped = True
        pred_label -= num_classes

    predicted_char = label_to_char.get(pred_label, '?')
    return predicted_char, flipped

# @app.route('/note', methods=['GET'])
# def test_model():
#     dummy_input = np.zeros((1, 28, 28, 1))  # Adjust shape for your model
#     prediction = model.predict(dummy_input)
#     return jsonify({"result": prediction.tolist()})


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        data = request.get_json()
        image_base64 = data.get('image')
        print("Received image data:", data.get('image')[:100])
        if not image_base64:
            return jsonify({'error': 'No image data received'}), 400

        

        image_tensor = preprocess_base64_image(image_base64)
        predicted_char, flipped = predict_and_correct(image_tensor)

        return jsonify({
            'prediction': predicted_char,
            'flipped': flipped
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
print("✅ Model loaded successfully")

if __name__ == '__main__':
    app.run(debug=True, port=8000)

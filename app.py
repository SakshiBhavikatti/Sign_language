from flask import Flask, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
model = load_model("sign_language_cnn_model.h5")
class_names = [chr(i) for i in range(65, 91) if chr(i) not in ['J', 'Z']]
TEST_FOLDER = "test_images"

def preprocess_image(image):
    image = image.convert('L') 
    image = image.resize((28, 28))  
    image = np.array(image).reshape(1, 28, 28, 1) / 255.0  
    return image

@app.route('/')
def predict_all_images():
    results = []

    for filename in os.listdir(TEST_FOLDER):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                path = os.path.join(TEST_FOLDER, filename)
                image = Image.open(path)
                processed = preprocess_image(image)
                prediction = model.predict(processed)
                predicted_index = int(np.argmax(prediction))
                confidence = float(np.max(prediction))
                predicted_class = class_names[predicted_index] if predicted_index < len(class_names) else str(predicted_index)

                results.append({
                    "filename": filename,
                    "predicted_class": predicted_class,
                    "confidence": round(confidence, 4)
                })
            except Exception as e:
                results.append({
                    "filename": filename,
                    "error": str(e)
                })

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import io
import tensorflow as tf
import keras
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

# Load the pre-trained model for image classification
model = keras.models.load_model("model.h5", compile=False)

# Initialize Flask app
app = Flask(__name__)

# Function to predict label for the given image

# Route to handle image prediction requests
@app.route("/predict", methods=["GET", "POST"])
def index():
    file = request.files.get('file')
    if file is None or file.filename == "":
        return jsonify({"error": "no file"})

    try:
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes))
        img = img.resize((224, 224), Image.NEAREST)
        img = img.convert('RGB')
        data = np.asarray(img)

        data = data / 255.0
        data = data.reshape(1, 224, 224, 3)
        prediction = model.predict(data)

        # Get the class names from the model
        class_names = [layer.name for layer in model.layers if layer.name.startswith('class_output')][0].weights[0].numpy()
        class_names = [class_name.decode('utf-8') for class_name in class_names]

        # Convert numpy array to Python list
        prediction = prediction.tolist()

        # Get the top-1 prediction and its corresponding class name
        top_prediction = np.argmax(prediction[0])
        top_class_name = class_names[top_prediction]

        data = {"prediction": prediction, "class_names": class_names, "top_class_name": top_class_name}
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__=="__main__":
    app.run(debug=True)
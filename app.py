from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import re, base64
from PIL import Image
from io import BytesIO

app = Flask(__name__)
CORS(app) 

model = tf.keras.models.load_model("digit_model.h5")
# loading our model 
def preprocess_image(img):
    img = img.convert("L")
    img = img.resize((28, 28))
    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['image']
    image_data = re.sub('^data:image/.+;base64,', '', data)
    img = Image.open(BytesIO(base64.b64decode(image_data)))
    
    processed = preprocess_image(img)
    pred = model.predict(processed)
    result = int(np.argmax(pred))
    
    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(debug=True)

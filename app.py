from flask import Flask, request, jsonify
import numpy as np
import cv2
import joblib

app = Flask(__name__)

model = joblib.load('model.pkl')

@app.route('/')
def home():
    return "Digit Predictor Running"

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28,28))
    img = 255 - img
    img = img / 255.0
    img = (img > 0.5).astype(int)
    
    img = img.reshape(1, -1)
    
    pred = model.predict(img)
    
    return jsonify({'prediction': int(pred[0])})

if __name__ == "__main__":
    app.run()

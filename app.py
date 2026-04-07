from flask import Flask, render_template, request
import numpy as np
import cv2
import pickle
import os

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

def preprocess_image(img):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    # Resize to 28x28 (MNIST format)
    img = cv2.resize(img, (28, 28))

    # Normalize
    img = img / 255.0

    # Flatten
    img = img.reshape(1, -1)

    return img

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        file = request.files["file"]
        file_path = "static/upload.png"
        file.save(file_path)

        img = preprocess_image(file_path)

        # Prediction
        pred = model.predict(img)[0]

        # Probability
        prob = model.predict_proba(img)
        confidence = np.max(prob)

        prediction = pred

    return render_template("index.html", prediction=prediction, confidence=confidence)

# IMPORTANT for deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
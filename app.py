from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Initialize Flask app
app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model
model = load_model("model/nail_disease_model.h5")
print("✅ Model loaded successfully")

# Class names (same order as training folders)
class_names = [
    "alopecia_areata",
    "beaus_lines",
    "bluish_nail",
    "clubbing",
    "darier_disease",
    "eczema",
    "half_and_half_nails",
    "koilonychia",
    "leukonychia",
    "muehrckes_lines",
    "onycholysis",
    "pale_nail",
    "red_lunula",
    "splinter_hemorrhage",
    "terrys_nail",
    "white_nail",
    "yellow_nails"
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files.get("image")

        if file:
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)

            # Load and preprocess image
            img = image.load_img(image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Predict
            preds = model.predict(img_array)
            predicted_index = np.argmax(preds)
            confidence = round(float(np.max(preds)) * 100, 2)
            prediction = class_names[predicted_index]

    # Notice: pass only "index.html", not "templates/index.html"
    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )

if __name__ == "__main__":
    app.run(debug=True)

# test.py
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("model/nail_disease_model.h5")
print("✅ Model loaded successfully")

img_path = "dataset/test/test/leukonychia/19.PNG"

if not os.path.exists(img_path):
    raise FileNotFoundError(f"❌ Image not found: {img_path}")

img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

predictions = model.predict(img_array)

predicted_index = np.argmax(predictions, axis=1)[0]
confidence = np.max(predictions) * 100

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

print("\n🩺 Predicted Disease:", class_names[predicted_index])
print("📊 Confidence:", round(confidence, 2), "%")

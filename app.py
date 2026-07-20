from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
import json
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load AI model
model = tf.keras.models.load_model("plant_disease_final.keras")

print("✅ AI Model Loaded Successfully!")


# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

print("✅ Class names loaded successfully!")


# Disease information
disease_info = {

    "Tomato___healthy": {
        "symptoms": "No disease detected. The plant appears healthy.",
        "treatment": "No treatment required. Continue regular care.",
        "prevention": "Provide proper watering, nutrition, and sunlight."
    },

    "Tomato___Early_blight": {
        "symptoms": "Brown spots with concentric rings on older leaves.",
        "treatment": "Apply a recommended fungicide and remove infected leaves.",
        "prevention": "Rotate crops and avoid overhead watering."
    },

    "Tomato___Late_blight": {
        "symptoms": "Large dark brown patches with white fungal growth.",
        "treatment": "Use fungicides immediately and remove infected plants.",
        "prevention": "Avoid excess moisture and improve air circulation."
    },

    "Tomato___Bacterial_spot": {
        "symptoms": "Small dark spots on leaves and fruits.",
        "treatment": "Apply copper-based bactericides.",
        "prevention": "Use disease-free seeds and avoid wet leaves."
    },

    "Tomato___Leaf_Mold": {
        "symptoms": "Yellow patches on upper leaves with mold underneath.",
        "treatment": "Apply fungicide and reduce humidity.",
        "prevention": "Increase ventilation in the field or greenhouse."
    },

    "Tomato___Septoria_leaf_spot": {
        "symptoms": "Small circular spots with gray centers.",
        "treatment": "Remove infected leaves and spray fungicide.",
        "prevention": "Avoid splashing water on leaves."
    },

    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "symptoms": "Tiny yellow spots and webbing on leaves.",
        "treatment": "Use miticides or insecticidal soap.",
        "prevention": "Maintain proper humidity and monitor plants regularly."
    },

    "Tomato___Target_Spot": {
        "symptoms": "Dark circular lesions on leaves.",
        "treatment": "Use fungicide and prune affected leaves.",
        "prevention": "Improve field sanitation."
    },

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "symptoms": "Yellow curled leaves and stunted growth.",
        "treatment": "Remove infected plants. There is no cure.",
        "prevention": "Control whiteflies and use resistant varieties."
    },

    "Tomato___Tomato_mosaic_virus": {
        "symptoms": "Mosaic pattern on leaves with curling.",
        "treatment": "Remove infected plants immediately.",
        "prevention": "Disinfect tools and avoid handling plants after tobacco products."
    }
}



# Home page
@app.route("/")
def home():
    return render_template("index.html")



# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "No file selected."

    file = request.files["file"]

    if file.filename == "":
        return "No image selected."


    # Save image
    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)


    # Preprocess image
    img = image.load_img(
        filepath,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = img_array / 255.0


    # Model prediction
    prediction = model.predict(img_array)


    predicted_index = np.argmax(prediction)

    confidence = float(
        np.max(prediction) * 100
    )


    disease = class_names[predicted_index]


    display_disease = disease.replace("___", " ").replace("_", " ")


    # Get disease details
    info = disease_info.get(
    disease,
    {
        "symptoms": "Information will be added soon.",
        "treatment": "Treatment information will be added soon.",
        "prevention": "Prevention information will be added soon."
    }
)


    return render_template(
        "result.html",
        disease=display_disease,
        confidence=f"{confidence:.2f}",
        image_path="/" + filepath,
        symptoms=info["symptoms"],
        treatment=info["treatment"],
        prevention=info["prevention"]
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
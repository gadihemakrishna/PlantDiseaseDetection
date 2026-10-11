
from flask import Flask, render_template, request, Response
import tensorflow as tf
import numpy as np
import os
import json
import requests
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = "plant_disease_final.keras"
CLASS_NAMES_PATH = "class_names.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"

disease_info = {
    "Tomato___Bacterial_spot": {
        "symptoms": "Small dark, water-soaked spots on leaves and fruit.",
        "treatment": "Remove infected leaves and use an appropriate treatment recommended by a local agricultural expert.",
        "prevention": "Avoid overhead watering, use clean tools, and rotate crops."
    },
    "Tomato___Early_blight": {
        "symptoms": "Brown leaf spots with concentric rings, often surrounded by yellowing.",
        "treatment": "Remove affected leaves and use a suitable fungicide according to its label.",
        "prevention": "Rotate crops, maintain spacing, and avoid wetting the leaves."
    },
    "Tomato___Late_blight": {
        "symptoms": "Dark, water-soaked lesions on leaves and stems that can spread rapidly.",
        "treatment": "Remove infected plant material and consult local agricultural guidance about appropriate disease control.",
        "prevention": "Improve airflow, avoid prolonged leaf wetness, and monitor plants regularly."
    },
    "Tomato___healthy": {
        "symptoms": "No obvious disease symptoms detected in the uploaded image.",
        "treatment": "No disease-specific treatment is indicated by this prediction.",
        "prevention": "Continue regular monitoring, proper watering, and balanced fertilization."
    }
}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No image was uploaded.", 400

    uploaded_file = request.files["file"]

    if not uploaded_file.filename:
        return "Please select an image.", 400

    if not allowed_file(uploaded_file.filename):
        return "Unsupported image type. Use PNG, JPG, JPEG, or WEBP.", 400

    filename = secure_filename(uploaded_file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        uploaded_file.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        predictions = model.predict(img_array, verbose=0)
        predicted_index = int(np.argmax(predictions[0]))
        disease_code = class_names[predicted_index]
        confidence = round(float(predictions[0][predicted_index]) * 100, 2)

        info = disease_info.get(
            disease_code,
            {
                "symptoms": "Disease information is not yet available for this class.",
                "treatment": "Confirm the diagnosis before applying any treatment. Consult a local agricultural expert.",
                "prevention": "Monitor the plant, maintain good hygiene, and avoid excessive leaf wetness."
            }
        )

        return render_template(
            "result.html",
            disease=disease_code.replace("___", " - ").replace("_", " "),
            disease_code=disease_code,
            confidence=confidence,
            image_path="/" + filepath.replace("\\", "/"),
            symptoms=info["symptoms"],
            treatment=info["treatment"],
            prevention=info["prevention"]
        )

    except Exception:
        app.logger.exception("Plant disease prediction failed")
        return "Unable to analyze this image. Please try another image.", 500


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.form.get("message", "").strip()
    disease = request.form.get("disease", "Unknown")

    if not user_message:
        return "Please enter a message.", 400

    payload = {
        "model": OLLAMA_MODEL,
        "stream": True,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are AgriVision AI, a helpful plant health assistant. "
                    f"The detected plant condition is {disease}. "
                    "Give clear, practical advice about plant diseases, "
                    "symptoms, treatment, and prevention. "
                    "Be concise and easy to understand. "
                    "Do not claim an image diagnosis is certainly correct."
                )
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    }

    try:
        ollama_response = requests.post(
            OLLAMA_URL,
            json=payload,
            stream=True,
            timeout=(10, 300)
        )
        ollama_response.raise_for_status()

    except requests.RequestException:
        app.logger.exception("Could not connect to Ollama")

        return (
            "Unable to connect to Ollama. Please check that Ollama is running.",
            503
        )

    def generate():
        try:
            for line in ollama_response.iter_lines():
                if not line:
                    continue

                data = json.loads(line.decode("utf-8"))
                content = data.get("message", {}).get("content", "")

                if content:
                    yield content

                if data.get("done", False):
                    break

        except Exception:
            app.logger.exception("Ollama streaming failed")
            yield "\n\nSorry, the response was interrupted."

        finally:
            ollama_response.close()

    return Response(
        generate(),
        mimetype="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

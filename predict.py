import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np


# Load trained model
model = tf.keras.models.load_model("plant_disease_model.h5")


# Disease class names (same order as dataset)
class_names = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


# Give your test image path here
img_path = "test_images\leaf.jpg"


# Load and preprocess image
img = image.load_img(
    img_path,
    target_size=(128,128)
)

img_array = image.img_to_array(img)

img_array = np.expand_dims(img_array, axis=0)

img_array = img_array / 255.0


# Prediction
prediction = model.predict(img_array)

predicted_class = np.argmax(prediction)


print("Prediction:")
print(class_names[predicted_class])

print("Confidence:")
print(round(np.max(prediction)*100,2), "%")
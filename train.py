import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os


# Dataset paths
train_path = "dataset/Tomato/train"
val_path = "dataset/Tomato/val"


# Image preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)


# Load training images
train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=(128,128),
    batch_size=32,
    class_mode="categorical"
)


# Load validation images
val_data = val_datagen.flow_from_directory(
    val_path,
    target_size=(128,128),
    batch_size=32,
    class_mode="categorical"
)


# CNN Model
model = Sequential([

    Conv2D(32,(3,3),activation="relu",input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64,(3,3),activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128,(3,3),activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(128,activation="relu"),
    Dropout(0.5),

    Dense(train_data.num_classes,activation="softmax")
])


# Compile model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# Show model details
model.summary()


# Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)


# Save trained model
model.save("plant_disease_model.h5")


print("Training Completed Successfully!")
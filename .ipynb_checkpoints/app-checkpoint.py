import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("tomato_model.h5")

# Class names in index order
class_names = [
    "Tomato__Bacterial_spot",
    "Tomato__Early_blight",
    "Tomato__Late_blight",
    "Tomato__Leaf_Mold",
    "Tomato__Septoria_leaf_spot",
    "Tomato__Spider_mites Two-spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__healthy"
]

img_size = (224, 224)

st.title("Smart Tomato Leaf Disease Detector")

uploaded = st.file_uploader("Upload a tomato leaf image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    img = image.resize(img_size)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)
    index = int(np.argmax(preds))
    confidence = float(np.max(preds))

    st.write(f"Prediction: *{class_names[index]}*")
    st.write(f"Confidence: {confidence:.2f}")
import streamlit as st
import numpy as np
import cv2
import os
import requests
from tensorflow.keras.models import load_model

# ================= CONFIG =================
IMAGE_SIZE = (180, 180)

MODEL_URL = "PASTE_MODEL_DIRECT_DOWNLOAD_LINK_HERE"
CLASSES_URL = "https://drive.google.com/file/d/1MbNajngKp_8Eiq0Bz1ieFs3JucSNiqww"

MODEL_PATH = "flower_cnn_model.h5"
CLASSES_PATH = "classes.npy"

# ================= DOWNLOAD UTILS =================
def download_file(url, path):
    if not os.path.exists(path):
        r = requests.get(url, stream=True)
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

# ================= DOWNLOAD FILES =================
download_file(MODEL_URL, MODEL_PATH)
download_file(CLASSES_URL, CLASSES_PATH)

# ================= LOAD MODEL =================
model = load_model(MODEL_PATH)
classes = np.load(CLASSES_PATH)

# ================= STREAMLIT UI =================
st.set_page_config(page_title="Flower Classification CNN", layout="centered")
st.title("🌸 Flower Classification App")
st.caption("CNN from scratch | Loaded via public Drive links")

uploaded_file = st.file_uploader(
    "Upload a flower image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = cv2.resize(img, IMAGE_SIZE)
    img_scaled = img_resized / 255.0
    img_input = np.expand_dims(img_scaled, axis=0)

    # Prediction
    preds = model.predict(img_input)
    class_index = np.argmax(preds)
    confidence = np.max(preds) * 100

    st.success(f"Prediction: {classes[class_index]}")
    st.write(f"Confidence: {confidence:.2f}%")

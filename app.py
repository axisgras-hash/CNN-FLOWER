import streamlit as st
import numpy as np
import os
import requests
from PIL import Image
from tensorflow.keras.models import load_model

# ================= CONFIG =================
IMAGE_SIZE = (180, 180)

MODEL_URL = "https://drive.google.com/uc?id=1MQyB9tui1VxR_qXZ_V2met0KuvUw16J-"
CLASSES_URL = "https://drive.google.com/uc?id=1MbNajngKp_8Eiq0Bz1ieFs3JucSNiqww"

MODEL_PATH = "flower_cnn_model.h5"
CLASSES_PATH = "classes.npy"

# ================= DOWNLOAD UTILS =================
def download_file(url, path):
    if not os.path.exists(path):
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

# ================= DOWNLOAD MODEL FILES =================
download_file(MODEL_URL, MODEL_PATH)
download_file(CLASSES_URL, CLASSES_PATH)

# ================= LOAD MODEL =================
model = load_model(MODEL_PATH)
classes = np.load(CLASSES_PATH)

# ================= STREAMLIT UI =================
st.set_page_config(page_title="Flower Classification CNN", layout="centered")

st.title("🌸 Flower Classification App")
st.caption("CNN from scratch | Streamlit Cloud compatible")

uploaded_file = st.file_uploader(
    "Upload a flower image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Load image safely using PIL
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image = image.resize(IMAGE_SIZE)
    image_array = np.array(image, dtype="float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    preds = model.predict(image_array)
    class_index = np.argmax(preds)
    confidence = np.max(preds) * 100

    st.success(f"Prediction: {classes[class_index]}")
    st.write(f"Confidence: {confidence:.2f}%")

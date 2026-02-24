import streamlit as st
import numpy as np
import pickle
import os
import gdown
from tensorflow.keras.models import load_model
from PIL import Image

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Flower Classification (CNN)", layout="centered")

st.title("🌸 Flower Classification using CNN")
st.write("Upload a flower image to predict its class")

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
IMG_SIZE = 150

MODEL_PATH = "flower_cnn.h5"
CLASSES_PATH = "classes.pkl"

# 🔗 GOOGLE DRIVE (DIRECT DOWNLOAD LINKS)
MODEL_URL = "https://drive.google.com/uc?id=1sn7Fv22Kq_XppVdbE0MWz1uxlY89wFNj"
CLASSES_URL = "https://drive.google.com/uc?id=1-juikjclAckEoUJfyDq9QTAKecPK4EXK"

# --------------------------------------------------
# DOWNLOAD FILES IF NOT PRESENT
# --------------------------------------------------
def download_file(url, output):
    if not os.path.exists(output):
        with st.spinner(f"Downloading {output}..."):
            gdown.download(url, output, quiet=False)

download_file(MODEL_URL, MODEL_PATH)
download_file(CLASSES_URL, CLASSES_PATH)

# --------------------------------------------------
# LOAD MODEL & CLASSES
# --------------------------------------------------
@st.cache_resource
def load_assets():
    model = load_model(MODEL_PATH, compile=False)
    with open(CLASSES_PATH, "rb") as f:
        classes = pickle.load(f)
    return model, classes

model, classes = load_assets()

# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Flower Image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=320)

    # Preprocess
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    # Output
    st.success(f"🌼 Predicted Flower: **{classes[class_index]}**")
    st.info(f"Confidence: **{confidence*100:.2f}%**")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown("""
**Made by:** Ankit Mishra  
**Role:** Data Science & AI Trainer
""")

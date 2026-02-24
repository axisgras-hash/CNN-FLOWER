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
st.set_page_config(
    page_title="Flower Classification (CNN)",
    layout="centered"
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("🌸 Flower Classification using CNN")
st.caption("Simple Convolutional Neural Network | Image Classification Demo")

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
IMG_SIZE = 150
CONF_THRESHOLD = 0.60

MODEL_PATH = "flower_cnn.h5"
CLASSES_PATH = "classes.pkl"

# 🔗 GOOGLE DRIVE (DIRECT DOWNLOAD LINKS – SAME AS YOURS)
MODEL_URL = "https://drive.google.com/uc?id=1YVmC1FYdRYcc_JsBhmOrRus3c7ACLlyu"
CLASSES_URL = "https://drive.google.com/uc?id=1-cXx7zF62mZgBQppxh8jHbGtxaquD66X"

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("ℹ️ About This App")
st.sidebar.markdown("""
**Model**
- Custom CNN (trained from scratch)

**Supported Classes**
- Daisy  
- Dandelion  
- Roses  
- Sunflowers  
- Tulips  

⚠️ Images outside these classes may give approximate results.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Use Case**
- CNN fundamentals  
- Image preprocessing  
- Confidence interpretation  
""")

# --------------------------------------------------
# DOWNLOAD FILES IF NOT PRESENT
# --------------------------------------------------
def download_file(url, output):
    if not os.path.exists(output):
        with st.spinner(f"Downloading {output} ..."):
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
    "📤 Upload a flower image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=320)

    # -------- Preprocessing --------
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -------- Prediction --------
    with st.spinner("Predicting..."):
        predictions = model.predict(img_array)[0]

    class_index = np.argmax(predictions)
    confidence = predictions[class_index]

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------
    st.subheader("✅ Prediction Result")

    if confidence < CONF_THRESHOLD:
        st.warning(
            f"Low confidence prediction.\n\n"
            f"Closest match: **{classes[class_index]}** "
            f"({confidence*100:.2f}%)"
        )
    else:
        st.success(f"🌼 **{classes[class_index]}**")
        st.info(f"Confidence: **{confidence*100:.2f}%**")

    # Confidence bar
    st.progress(float(confidence))

    # --------------------------------------------------
    # TOP-3 PREDICTIONS (DYNAMIC PART)
    # --------------------------------------------------
    st.subheader("📊 Top-3 Predictions")

    top_indices = np.argsort(predictions)[::-1][:3]

    for i in top_indices:
        st.write(f"**{classes[i]}** — {predictions[i]*100:.2f}%")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown("""
**Made by:** Ankit Mishra  
**Role:** Data Science & AI Trainer
""")

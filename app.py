import streamlit as st
import numpy as np
import os
import requests
from PIL import Image
from tensorflow.keras.models import load_model

# =====================================================
# CONFIG
# =====================================================
IMAGE_SIZE = (180, 180)

MODEL_URL = "https://drive.google.com/uc?id=1MQyB9tui1VxR_qXZ_V2met0KuvUw16J-"
CLASSES_URL = "https://drive.google.com/uc?id=1MbNajngKp_8Eiq0Bz1ieFs3JucSNiqww"

MODEL_PATH = "flower_cnn_model.h5"
CLASSES_PATH = "classes.npy"

BANNER_IMAGE_URL = (
    "https://images.unsplash.com/photo-1501004318641-b39e6451bec6"
)

# =====================================================
# DOWNLOAD UTIL
# =====================================================
def download_file(url, path):
    if not os.path.exists(path):
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

# =====================================================
# LOAD MODEL & CLASSES
# =====================================================
download_file(MODEL_URL, MODEL_PATH)
download_file(CLASSES_URL, CLASSES_PATH)

model = load_model(MODEL_PATH)
classes = np.load(CLASSES_PATH)

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Flower Classification using CNN",
    page_icon="🌸",
    layout="centered"
)

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("🌸 CNN Flower Classifier")

st.sidebar.markdown("""
### 📌 Project Overview
This project demonstrates a **Convolutional Neural Network (CNN)** built **from scratch**
to classify flower images into multiple categories.

**Classes used:**
- Daisy  
- Dandelion  
- Roses  
- Sunflowers  
- Tulips  

---

### 🧠 Model Highlights
- Deep CNN architecture  
- Batch Normalization  
- Dropout to reduce overfitting  
- Image Augmentation  
- Trained without any pretrained models  

---

### 🛠 Tech Stack
- Python  
- TensorFlow / Keras  
- NumPy  
- Streamlit  
- PIL  

---

### 🎯 Use Case
- Computer Vision learning  
- CNN fundamentals  
- ML deployment demo  
- Student & portfolio project  
""")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Trainer:** Ankit Mishra  \n"
    "**Domain:** AI / ML / Data Science"
)

# =====================================================
# MAIN UI
# =====================================================
st.markdown(
    "<h1 style='text-align: center;'>🌸 Flower Classification using CNN</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Upload a flower image and let the CNN predict its class</p>",
    unsafe_allow_html=True
)

# Banner Image
st.image(BANNER_IMAGE_URL, use_column_width=True)

st.markdown("---")

# =====================================================
# FILE UPLOADER
# =====================================================
uploaded_file = st.file_uploader(
    "📤 Upload a flower image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocessing
    image_resized = image.resize(IMAGE_SIZE)
    image_array = np.array(image_resized, dtype="float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    preds = model.predict(image_array)
    class_index = np.argmax(preds)
    confidence = np.max(preds) * 100

    st.markdown("### 🔍 Prediction Result")
    st.success(f"**Flower Type:** {classes[class_index]}")
    st.info(f"**Confidence:** {confidence:.2f}%")

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")

st.markdown(
    """
    <div style="text-align: center; font-size: 14px; color: gray;">
        Built with ❤️ using CNN & Streamlit <br>
        © 2026 | AI & Data Science Project <br>
        Trainer: <b>Ankit Mishra</b>
    </div>
    """,
    unsafe_allow_html=True
)

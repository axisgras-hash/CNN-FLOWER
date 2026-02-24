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

FLOWER_EMOJI = {
    "daisy": "🌼",
    "dandelion": "🌾",
    "roses": "🌹",
    "sunflowers": "🌻",
    "tulips": "🌷"
}

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
    page_title="Flower Classification | CNN",
    page_icon="🌸",
    layout="centered"
)

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("🌸 Flower Classifier")

st.sidebar.markdown("""
### 📌 About Project
This application classifies flower images using a **Convolutional Neural Network (CNN)** 
trained **from scratch**.

It demonstrates the **complete ML lifecycle**:
- Image preprocessing  
- CNN-based feature learning  
- Real-time inference  
- Web deployment  

---

### 🧠 Model Details
- Custom CNN architecture  
- Data augmentation  
- Dropout for regularization  
- No pretrained models  

---

### 🌼 Flower Classes
- 🌼 Daisy  
- 🌾 Dandelion  
- 🌹 Roses  
- 🌻 Sunflowers  
- 🌷 Tulips  

---

### 🛠 Tech Stack
- Python  
- TensorFlow / Keras  
- Streamlit  
- NumPy  
- PIL  
""")

st.sidebar.markdown("---")
st.sidebar.markdown("👨‍🏫 **Trainer:** Ankit Mishra")

# =====================================================
# MAIN UI
# =====================================================
st.markdown(
    "<h1 style='text-align:center;'>🌸 Flower Classification using CNN</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>Upload a flower image and click <b>Predict</b></p>",
    unsafe_allow_html=True
)

st.markdown("---")

# =====================================================
# FILE UPLOADER
# =====================================================
uploaded_file = st.file_uploader(
    "📤 Upload a flower image",
    type=["jpg", "jpeg", "png","webp"]
)

# =====================================================
# PREDICTION FLOW
# =====================================================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.markdown("")  # spacing

    if st.button("🔍 Predict Flower"):
        with st.spinner("Analyzing image... 🌸"):
            image_resized = image.resize(IMAGE_SIZE)
            image_array = np.array(image_resized, dtype="float32") / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            preds = model.predict(image_array)
            class_index = np.argmax(preds)
            confidence = np.max(preds) * 100

            flower_name = classes[class_index]
            emoji = FLOWER_EMOJI.get(flower_name, "🌸")

        st.success(f"### {emoji} Prediction: **{flower_name.capitalize()}**")
        st.info(f"### 📊 Confidence: **{confidence:.2f}%**")

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; font-size:13px; color:gray;">
        🌸 CNN Image Classification Project <br>
        Built with TensorFlow & Streamlit <br>
        © 2026 | Trainer: <b>Ankit Mishra</b>
    </div>
    """,
    unsafe_allow_html=True
)

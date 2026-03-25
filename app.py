import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# =========================
# CLASS NAMES (EDIT IF NEEDED)
# =========================
class_names = [
    "Healthy",
    "Early Blight",
    "Late Blight",
    "Leaf Mold"
]

# =========================
# LOAD MODEL (.h5 ONLY)
# =========================
@st.cache_resource
def load_model_safe():
    try:
        model = tf.keras.models.load_model("best_model.h5", compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Model Load Failed: {e}")
        return None

model = load_model_safe()

# =========================
# IMAGE PREPROCESSING
# =========================
def preprocess_image(image):
    image = image.resize((224, 224))
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# =========================
# PREDICTION FUNCTION
# =========================
def predict(image):
    if model is None:
        return "Model Not Loaded", 0.0

    processed = preprocess_image(image)
    preds = model.predict(processed)

    class_index = np.argmax(preds)
    label = class_names[class_index]
    confidence = float(np.max(preds))

    return label, confidence

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="CropDoctor AI", page_icon="🌿")

st.title("🌿 CropDoctor AI")

uploaded_file = st.file_uploader("Upload Crop Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        label, conf = predict(image)

        st.success(f"🦠 Disease: {label}")
        st.info(f"📊 Confidence: {conf:.3f}")
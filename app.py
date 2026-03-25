import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import datetime

# =========================
# CLASS NAMES (ALL CLASSES)
# =========================
class_names = [
'Alternaria_D','Botrytis Leaf Blight','Bulb Rot','Bulb_blight-D','Caterpillar-P',
'Downy mildew','Fusarium-D','Healthy leaves','Iris yellow virus_augment',
'Pepper__bell___Bacterial_spot','Pepper__bell___healthy','Potato___Early_blight',
'Potato___Late_blight','Potato___healthy','Purple blotch','Rust',
'Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_Late_blight',
'Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot',
'Tomato_Spider_mites_Two_spotted_spider_mite','Tomato__Target_Spot',
'Tomato__Tomato_YellowLeaf__Curl_Virus','Tomato__Tomato_mosaic_virus',
'Tomato_healthy','Virosis-D','Xanthomonas Leaf Blight','onion1',
'stemphylium Leaf Blight'
]

# =========================
# LOAD CNN MODEL
# =========================
try:
    image_model = tf.keras.models.load_model("best_model.h5")
    st.success("✅ CNN Model Loaded")
except Exception as e:
    st.error(f"❌ CNN Load Failed: {e}")
    image_model = None

# =========================
# IMAGE PREDICTION
# =========================
def predict_image(image):
    if image_model is None:
        return "Model Not Loaded", 0.5

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = image_model.predict(img_array)
    class_index = np.argmax(preds)
    confidence = float(np.max(preds))

    disease = class_names[class_index].replace("_", " ")
    return disease, confidence

# =========================
# UI
# =========================
st.title("🌱 CropDoctor - Multimodal AI System")

image_file = st.file_uploader("Upload Crop Image", type=["jpg", "png", "jpeg"])

date = st.date_input("Date", datetime.date.today())
temperature = st.number_input("Temperature (°C)", 0.0)
humidity = st.number_input("Humidity (%)", 0.0)
rainfall = st.number_input("Rainfall (mm)", 0.0)
prev_risk = st.number_input("Previous Risk Index", 0.0)

# =========================
# MAIN BUTTON
# =========================
if st.button("Analyze Crop"):

    if image_file is None:
        st.error("Please upload an image")
    else:
        with st.spinner("Analyzing crop..."):

            image = Image.open(image_file)

            # =========================
            # 1. CNN PREDICTION
            # =========================
            disease, confidence = predict_image(image)

            # =========================
            # 2. ENVIRONMENTAL LOGIC (REPLACED RF)
            # =========================
            env_raw = (temperature + humidity + rainfall + prev_risk) / 4

            if env_raw < 30:
                env_label = "Low"
            elif env_raw < 70:
                env_label = "Medium"
            else:
                env_label = "High"

            env_risk = env_raw / 100

            # =========================
            # 3. SIMPLE FORECAST (REPLACED ARIMA SAFE)
            # =========================
            forecast_risk = env_risk * 0.95 + 0.05  # small variation

            # =========================
            # 4. FUSION MODEL
            # =========================
            final_score = (
                0.6 * confidence +
                0.25 * env_risk +
                0.15 * forecast_risk
            )

            if final_score < 0.3:
                final_level = "🟢 Low Risk"
                action = "Crop is healthy. Maintain regular monitoring."
            elif final_score < 0.7:
                final_level = "⚠ Medium Risk"
                action = "Monitor crop closely and apply preventive treatment."
            else:
                final_level = "⚠ High Risk"
                action = "High disease risk. Apply pesticides immediately."

        # =========================
        # OUTPUT
        # =========================
        st.markdown("## 📊 CROP DOCTOR FINAL REPORT")

        st.image(image, caption="Uploaded Image")

        st.write(f"**Detected Disease:** {disease}")
        st.write(f"**Image Confidence:** {confidence:.3f}")
        st.write(f"**Environmental Risk:** {env_label}")
        st.write(f"**Forecast Risk Score:** {forecast_risk:.3f}")
        st.write(f"**Final Risk Score:** {final_score:.3f}")
        st.write(f"**Final Alert Level:** {final_level}")
        st.write(f"**Recommended Action:** {action}")

# =========================
# FOOTER
# =========================
st.caption("CropDoctor v1.0 - Multimodal AI System")

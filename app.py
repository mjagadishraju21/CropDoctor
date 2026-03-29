import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image
import datetime
import os
import traceback

# =========================
# CLASS NAMES
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
# LOAD MODELS (SAFE)
# =========================

# RF MODEL
rf_model = None
rf_path = "crop_doctor_rf_model.pkl"

if os.path.exists(rf_path):
    try:
        rf_model = joblib.load(rf_path)
        st.success("✅ RF Model Loaded")
    except Exception as e:
        st.error("❌ RF Model failed to load")
        st.text(str(e))
        traceback.print_exc()
        rf_model = None
else:
    st.warning("⚠ RF Model not found")

# ARIMA MODEL
arima_model = None
arima_path = "crop_doctor_arima_model.pkl"

if os.path.exists(arima_path):
    try:
        import statsmodels.api as sm
        arima_model = joblib.load(arima_path)
        st.success("✅ ARIMA Model Loaded")
    except Exception as e:
        st.error("❌ ARIMA Model failed to load")
        st.text(str(e))
        arima_model = None
else:
    st.warning("⚠ ARIMA Model not found")

# CNN MODEL
image_model = None
cnn_path = "best_model.h5"

if os.path.exists(cnn_path):
    try:
        image_model = tf.keras.models.load_model(cnn_path)
        st.success("✅ CNN Model Loaded")
    except Exception as e:
        st.error("❌ CNN Model failed to load")
        st.text(str(e))
        image_model = None
else:
    st.error("❌ CNN Model Missing")

# =========================
# UI
# =========================

st.title("🌱 CropDoctor - Multimodal AI System")
st.write("Upload crop image + enter environmental conditions")

# INPUTS
image_file = st.file_uploader("Upload Crop Image", type=["jpg", "png", "jpeg"])

date = st.date_input("Date", datetime.date.today())
temperature = st.number_input("Temperature (°C)", 0.0)
humidity = st.number_input("Humidity (%)", 0.0)
rainfall = st.number_input("Rainfall (mm)", 0.0)
prev_risk = st.number_input("Previous Risk Index", 0.0)

# =========================
# ANALYZE BUTTON
# =========================

if st.button("Analyze Crop"):

    if image_file is None:
        st.error("Please upload an image")
    else:

        # =========================
        # IMAGE MODEL
        # =========================
        if image_model:
            try:
                image = Image.open(image_file).resize((224, 224))
                img_array = np.array(image) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                preds = image_model.predict(img_array)
                class_index = int(np.argmax(preds))
                confidence = float(np.max(preds))

                if class_index >= len(class_names):
                    class_index = 0

                disease = class_names[class_index].replace("_", " ")

            except Exception as e:
                st.error("❌ Image processing failed")
                st.text(str(e))
                disease = "Error"
                confidence = 0.0
        else:
            disease = "Model Not Loaded"
            confidence = 0.5

        # =========================
        # ENVIRONMENT MODEL
        # =========================
        if rf_model:
            try:
                date_num = date.toordinal()
                env_input = np.array([[date_num, temperature, humidity, rainfall, prev_risk]])
                env_risk = float(rf_model.predict(env_input)[0])

                if env_risk < 0.3:
                    env_label = "Low"
                elif env_risk < 0.7:
                    env_label = "Medium"
                else:
                    env_label = "High"
            except:
                env_risk = 0.5
                env_label = "Error"
        else:
            env_risk = 0.5
            env_label = "Unknown"

        # =========================
        # FORECAST MODEL
        # =========================
        if arima_model:
            try:
                forecast = arima_model.forecast(steps=1)
                forecast_risk = float(forecast[0])
            except:
                forecast_risk = env_risk
        else:
            forecast_risk = env_risk

        # =========================
        # FUSION
        # =========================
        final_score = (
            0.5 * confidence +
            0.3 * env_risk +
            0.2 * forecast_risk
        )

        # FINAL DECISION
        if final_score < 0.3:
            final_level = "🟢 Low Risk"
            action = "Crop is healthy. Maintain regular monitoring."
        elif final_score < 0.7:
            final_level = "⚠ Medium Risk"
            action = "Monitor crop closely and apply preventive treatment."
        else:
            final_level = "🚨 High Risk"
            action = "High disease risk. Apply pesticides immediately."

        # =========================
        # OUTPUT
        # =========================
        st.markdown("## 📊 CROP DOCTOR FINAL REPORT")

        st.write(f"**Detected Disease:** {disease}")
        st.write(f"**Image Confidence:** {confidence:.3f}")
        st.write(f"**Environmental Risk:** {env_label}")
        st.write(f"**Forecast Risk Score:** {forecast_risk:.3f}")
        st.write(f"**Final Risk Score:** {final_score:.3f}")
        st.write(f"**Final Alert Level:** {final_level}")
        st.write(f"**Recommended Action:** {action}")
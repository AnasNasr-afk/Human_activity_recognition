import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# ============================
# LOAD SAVED OBJECTS
# ============================
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
label_encoder = joblib.load("label_encoder.pkl")

log_reg = joblib.load("logistic_regression.pkl")
rf = joblib.load("random_forest.pkl")

# ============================
# ACTIVITY LABEL NAMES (UCI HAR)
# ============================
activity_names = {
    0: "WALKING",
    1: "WALKING_UPSTAIRS",
    2: "WALKING_DOWNSTAIRS",
    3: "SITTING",
    4: "STANDING",
    5: "LAYING"
}

# ============================
# UI HEADER
# ============================
st.title("Human Activity Recognition (HAR)")
st.write("Prediction using pre-extracted features")

# ============================
# MODEL SELECTION
# ============================
model_name = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Random Forest"]
)

model = log_reg if model_name == "Logistic Regression" else rf

# ============================
# CLEAR PREDICTION DISPLAY
# ============================
def display_prediction(activity, confidence):
    st.markdown("---")
    st.markdown("## 🔍 Final Prediction")
    st.markdown(
        f"<h1 style='color:#00ff99; text-align:center;'>{activity}</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<h3 style='text-align:center;'>📊 Confidence: {confidence:.2f}%</h3>",
        unsafe_allow_html=True
    )
    st.markdown("---")

# ============================
# INPUT MODE SELECTION
# ============================
input_mode = st.radio(
    "Choose Input Option",
    [
        "Upload CSV of Features",
        "Simulated Sensor Data (Real-Time)",
        "Manual Sliders for Key Features"
    ]
)

# ============================================================
# INPUT OPTION 1: UPLOAD CSV OF FEATURES
# ============================================================
if input_mode == "Upload CSV of Features":
    st.subheader("Input Option 1: Upload CSV of Features")

    uploaded_file = st.file_uploader(
        "Upload a CSV file containing pre-extracted features",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            X_scaled = scaler.transform(df)
            X_pca = pca.transform(X_scaled)

            preds = model.predict(X_pca)
            probs = model.predict_proba(X_pca)

            activity_id = int(preds[0])
            activity = activity_names.get(activity_id, str(activity_id))
            confidence = np.max(probs[0]) * 100

            display_prediction(activity, confidence)

        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")

# ============================================================
# INPUT OPTION 2: SIMULATED SENSOR DATA (REAL-TIME)
# ============================================================
elif input_mode == "Simulated Sensor Data (Real-Time)":
    st.subheader("Input Option 2: Simulated Sensor Data (Real-Time)")

    st.write(
        "This mode simulates streaming sensor data. "
        "Predictions update automatically in real time."
    )

    placeholder = st.empty()

    if st.button("Start Simulation"):
        for _ in range(20):
            simulated_features = np.random.normal(
                loc=0.0,
                scale=1.0,
                size=pca.n_components_
            )

            X_pca_sim = pd.DataFrame([simulated_features])

            pred = model.predict(X_pca_sim)
            prob = model.predict_proba(X_pca_sim)

            activity_id = int(pred[0])
            activity = activity_names.get(activity_id, str(activity_id))
            confidence = np.max(prob[0]) * 100

            with placeholder.container():
                display_prediction(activity, confidence)

            time.sleep(0.5)

# ============================================================
# INPUT OPTION 3: MANUAL SLIDERS (REAL-TIME)
# ============================================================
elif input_mode == "Manual Sliders for Key Features":
    st.subheader("Input Option 3: Manual Feature Input (Real-Time)")

    st.write(
        "Adjust the sliders below to simulate real-time feature changes. "
        "Predictions update instantly."
    )

    num_pca_features = pca.n_components_

    slider_features = []
    for i in range(min(10, num_pca_features)):
        value = st.slider(
            f"PCA Feature {i + 1}",
            min_value=-5.0,
            max_value=5.0,
            value=0.0,
            step=0.1,
            key=f"slider_{i}"
        )
        slider_features.append(value)

    # Add small noise to remaining PCA features (prevents class bias)
    slider_features += list(
        np.random.normal(0, 0.2, num_pca_features - len(slider_features))
    )

    X_pca_input = pd.DataFrame([slider_features])

    pred = model.predict(X_pca_input)
    prob = model.predict_proba(X_pca_input)

    activity_id = int(pred[0])
    activity = activity_names.get(activity_id, str(activity_id))
    confidence = np.max(prob[0]) * 100

    display_prediction(activity, confidence)

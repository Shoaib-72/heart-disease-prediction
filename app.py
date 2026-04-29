import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ---------------- Load Model ----------------
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- Custom CSS ----------------
st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa;
        }

        .title {
            text-align: center;
            color: #d63384;
            font-size: 42px;
            font-weight: bold;
            margin-bottom: 10px;
        }

        .subtitle {
            text-align: center;
            color: #555;
            font-size: 18px;
            margin-bottom: 30px;
        }

        .card {
            background-color: white;
            padding: 25px;
            border-radius: 20px;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }

        .result-box {
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
        }

        .healthy {
            background-color: #d4edda;
            color: #155724;
        }

        .risk {
            background-color: #f8d7da;
            color: #721c24;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown('<div class="title">❤️ Heart Disease Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict the possibility of heart disease using patient medical details</div>', unsafe_allow_html=True)

# ---------------- Layout ----------------
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Patient Information")

    c1, c2 = st.columns(2)

    with c1:
        age = st.slider("Age", 20, 100, 45)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
        trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
        chol = st.slider("Cholesterol", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
        restecg = st.selectbox("Resting ECG Result", [0, 1, 2])

    with c2:
        thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
        oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
        slope = st.selectbox("Slope", [1, 2, 3])
        ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia", [3, 6, 7])

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Entered Values")

    sex_value = 1 if sex == "Male" else 0

    patient_data = {
        "Age": age,
        "Sex": sex,
        "Chest Pain": cp,
        "BP": trestbps,
        "Cholesterol": chol,
        "FBS": fbs,
        "ECG": restecg,
        "Max HR": thalach,
        "Angina": exang,
        "Oldpeak": oldpeak,
        "Slope": slope,
        "CA": ca,
        "Thal": thal
    }

    st.dataframe(pd.DataFrame(patient_data.items(), columns=["Feature", "Value"]), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Prediction ----------------
if st.button("Predict Heart Disease Risk"):

    input_df = pd.DataFrame([{
    'Age': age,
    'Sex': sex_value,
    'Chest pain type': cp,
    'BP': trestbps,
    'Cholesterol': chol,
    'FBS over 120': fbs,
    'EKG results': restecg,
    'Max HR': thalach,
    'Exercise angina': exang,
    'ST depression': oldpeak,
    'Slope of ST': slope,
    'Number of vessels fluro': ca,
    'Thallium': thal
}])

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1] * 100

    result = int(prediction[0])   # ✅ FIX

    st.markdown("<br>", unsafe_allow_html=True)

    if result == 1:
        st.markdown(
            f'<div class="result-box risk">⚠️ High Risk of Heart Disease<br><br>Risk Probability: {probability:.2f}%</div>',
            unsafe_allow_html=True
        )

        st.warning("""
        - Maintain a healthy diet  
        - Exercise regularly  
        - Reduce stress  
        - Avoid smoking and alcohol  
        - Consult a cardiologist  
        """)

    else:
        st.markdown(
            f'<div class="result-box healthy">✅ Low Risk of Heart Disease<br><br>Risk Probability: {probability:.2f}%</div>',
            unsafe_allow_html=True
        )

        st.success("""
        - Continue healthy eating habits  
        - Stay physically active  
        - Maintain regular health checkups  
        - Sleep well and manage stress  
        """)

    st.progress(int(probability))
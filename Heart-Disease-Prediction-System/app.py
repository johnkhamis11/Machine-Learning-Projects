import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from catboost import CatBoostClassifier

st.set_page_config(
    page_title="Heart Health AI",
    page_icon="❤️",
    layout="wide"
)

@st.cache_resource
def load_assets():
    model = pickle.load(open("heart_disease_catboost_model.pkl", "rb"))
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_assets()
except:
    st.error("Model or Scaler files not found!")

st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        border: none;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; color:#ff4b4b; font-family:sans-serif;'>❤️ Heart Disease Diagnostics</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#555;'>Advanced Machine Learning Prediction System</p>", unsafe_allow_html=True)
st.divider()

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    with st.expander("👤 Patient Demographics", expanded=True):
        c1, c2 = st.columns(2)
        age = c1.number_input("Age", 1, 120, 45)
        sex_raw = c2.selectbox("Gender", ["Male", "Female"])
        sex = 1 if sex_raw == "Male" else 0

    with st.expander("🩺 Clinical Measurements", expanded=True):
        c3, c4, c5 = st.columns(3)
        cp = c3.selectbox("Chest Pain (0-3)", [0, 1, 2, 3])
        trestbps = c4.number_input("Resting BP (mmHg)", 50, 250, 120)
        chol = c5.number_input("Cholesterol (mg/dl)", 100, 600, 200)
        
        c6, c7, c8 = st.columns(3)
        fbs = c6.selectbox("Fasting Sugar > 120", ["No", "Yes"])
        fbs = 1 if fbs == "Yes" else 0
        restecg = c7.selectbox("Resting ECG (0-2)", [0, 1, 2])
        thalach = c8.number_input("Max Heart Rate", 50, 220, 150)

    with st.expander("🔬 Specialized Tests", expanded=True):
        c9, c10, c11 = st.columns(3)
        exang = c9.selectbox("Exercise Angina", ["No", "Yes"])
        exang = 1 if exang == "Yes" else 0
        oldpeak = c10.number_input("ST Depression", 0.0, 10.0, 1.0)
        slope = c11.selectbox("ST Slope (0-2)", [0, 1, 2])
        
        c12, c13 = st.columns(2)
        ca = c12.selectbox("Major Vessels (0-4)", [0, 1, 2, 3, 4])
        thal = c13.selectbox("Thalassemia (0-3)", [0, 1, 2, 3])

    if age < 40:
        age_group = 0
    elif age < 55:
        age_group = 1
    else:
        age_group = 2

    features = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, age_group]],
                            columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "age_group"])

    st.write("")
    if st.button("RUN ANALYSIS"):
        scaled_data = scaler.transform(features)
        prediction = model.predict(scaled_data)
        probability = model.predict_proba(scaled_data)[0][1]

        st.divider()
        
        if prediction[0] == 1:
            st.markdown(f"""
                <div style="background-color:#ffecec; padding:20px; border-radius:10px; border-left: 8px solid #ff4b4b;">
                    <h2 style="color:#ff4b4b; margin:0;">High Risk Detected</h2>
                    <p style="color:#333;">The model indicates a high probability of heart disease ({probability:.1%}). Please consult a specialist.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="background-color:#eaffed; padding:20px; border-radius:10px; border-left: 8px solid #28a745;">
                    <h2 style="color:#28a745; margin:0;">Low Risk Detected</h2>
                    <p style="color:#333;">The model suggests a low risk of heart disease ({1-probability:.1%} confidence). Maintain a healthy lifestyle!</p>
                </div>
            """, unsafe_allow_html=True)

st.markdown("<p style='text-align:center; color:gray; margin-top:50px;'>Note: This tool is for educational purposes only.</p>", unsafe_allow_html=True)
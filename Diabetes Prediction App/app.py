import streamlit as st
import joblib
import numpy as np

# Load resources
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Page Configuration
st.set_page_config(page_title="Diabetes Prediction System", page_icon="🩺", layout="wide")

# Custom CSS for the result box
st.markdown("""
    <style>
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 24px;
        margin-top: 20px;
    }
    .positive { background-color: #ff4b4b; color: white; }
    .negative { background-color: #245336; color: #ffffff; }
    </style>
    """, unsafe_allow_html=True)

# Header Section
st.title("🩺 Diabetes Prediction System")
st.write("Enter the patient's clinical data below to predict the likelihood of diabetes.")

# Input Form
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, step=1, value=3)
        glucose = st.number_input("Glucose Level", min_value=0, value=80)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, value=60)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, value=50)
        
    with col2:
        insulin = st.number_input("Insulin Level", min_value=0, value=40)
        bmi = st.number_input("BMI", min_value=0.0, format="%.1f", value=35.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f", value=0.670)
        age = st.number_input("Age", min_value=1, value=30)

st.markdown("---")

# Prediction Logic
if st.button("Predict Result"):
    # Prepare features
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                          insulin, bmi, dpf, age]])
    
    # Scaling and Prediction
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    probability = model.predict_proba(scaled_features)
    confidence = np.max(probability) * 100

    # Display Styled Result
    if prediction[0] == 1:
        st.markdown(f'<div class="result-box positive">Result: Diabetic (Confidence: {confidence:.2f}%)</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-box negative">Result: Non-Diabetic (Confidence: {confidence:.2f}%)</div>', unsafe_allow_html=True)
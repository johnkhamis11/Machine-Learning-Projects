import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('breast_cancer_adaboost_final.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Breast Cancer Detector", page_icon="⚕️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    .stNumberInput { border-radius: 8px; }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3.5em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover { background-color: #0056b3; border: none; }
    .result-card {
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #004085;'>🩺 Breast Cancer Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #495057;'>Professional Diagnostic Support System | Medical Analysis Edition ⚕️</p>", unsafe_allow_html=True)
st.divider()

selected_features = [
    'area_worst', 'concave points_worst', 'texture_worst', 'concave points_mean',
    'area_se', 'radius_worst', 'texture_mean', 'concavity_worst', 
    'smoothness_worst', 'area_mean'
]

all_features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
    'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst',
    'symmetry_worst', 'fractal_dimension_worst'
]

clinical_defaults = {
    'radius_mean': 14.12, 'texture_mean': 19.28, 'perimeter_mean': 91.96, 'area_mean': 654.88,
    'smoothness_mean': 0.096, 'compactness_mean': 0.104, 'concavity_mean': 0.088,
    'concave points_mean': 0.048, 'symmetry_mean': 0.181, 'fractal_dimension_mean': 0.062,
    'radius_se': 0.405, 'texture_se': 1.216, 'perimeter_se': 2.866, 'area_se': 40.33,
    'smoothness_se': 0.007, 'compactness_se': 0.025, 'concavity_se': 0.031,
    'concave points_se': 0.011, 'symmetry_se': 0.020, 'fractal_dimension_se': 0.003,
    'radius_worst': 16.26, 'texture_worst': 25.67, 'perimeter_worst': 107.26, 'area_worst': 880.58,
    'smoothness_worst': 0.132, 'compactness_worst': 0.254, 'concavity_worst': 0.272,
    'concave points_worst': 0.114, 'symmetry_worst': 0.290, 'fractal_dimension_worst': 0.083
}

st.subheader("📊 Clinical Parameters Extraction")
input_data = {}
cols = st.columns(2)

for i, feature in enumerate(selected_features):
    col = cols[i % 2]
    default_val = clinical_defaults.get(feature, 0.0)
    input_data[feature] = col.number_input(
        f"💉 {feature.replace('_', ' ').title()}", 
        value=float(default_val), 
        format="%.4f"
    )

st.write("")
if st.button("🔍 EXECUTE DIAGNOSTIC SCAN"):
    full_input_dict = clinical_defaults.copy()
    full_input_dict.update(input_data)
    input_df = pd.DataFrame([full_input_dict])[all_features]
    
    scaled_data = scaler.transform(input_df)
    prediction = model.predict(scaled_data)[0]
    probabilities = model.predict_proba(scaled_data)[0]

    st.divider()
    
    res_col1, res_col2 = st.columns([2, 1])
    
    if prediction == 1:
        with res_col1:
            st.markdown(f"""
                <div class="result-card" style="background-color: #fdf2f2; border-top: 5px solid #dc3545;">
                    <h2 style="color: #dc3545;">⚠️ Detection: Malignant</h2>
                    <p style="color: #495057;">The system has detected malignant cellular characteristics. Clinical intervention is required.</p>
                </div>
            """, unsafe_allow_html=True)
        res_col2.metric("Probability Score", f"{probabilities[1]:.2%}", "Critical", delta_color="inverse")
    else:
        with res_col1:
            st.markdown(f"""
                <div class="result-card" style="background-color: #f0f9f1; border-top: 5px solid #28a745;">
                    <h2 style="color: #28a745;">🟢 Detection: Benign</h2>
                    <p style="color: #495057;">The analysis confirms benign cellular growth. No immediate malignancy detected.</p>
                </div>
            """, unsafe_allow_html=True)
        res_col2.metric("Probability Score", f"{probabilities[0]:.2%}", "Stable")

st.markdown("<br><hr><p style='text-align: center; font-size: 0.8rem; color: #6c757d;'>🏥 Medical Diagnostic Intelligence | Data-Driven Health Analysis</p>", unsafe_allow_html=True)
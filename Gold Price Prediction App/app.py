import streamlit as st
import numpy as np
import joblib

# 1. Page Configuration
st.set_page_config(
    page_title="Gold Price Analytics",
    page_icon="📈",
    layout="centered"
)

# 2. Custom CSS for a clean, modern look
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        border: none;
    }
    .prediction-box {
        background-color: #f8f9fa;
        padding: 30px;
        border-radius: 15px;
        border: 1px solid #dee2e6;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Load model and scaler
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("best_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None

model, scaler = load_assets()

# 4. Header
st.title("📈 Gold Price Forecasting")
st.markdown("Enter the market variables below to generate a real-time price prediction.")
st.divider()

# 5. Input Section
if model and scaler:
    st.subheader("Input Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        spx = st.number_input("SPX Index", value=1400.0, step=1.0)
        gld = st.number_input("GLD Entry", value=100.0, step=0.1)
        uso = st.number_input("USO (Oil)", value=30.0, step=0.1)
    
    with col2:
        slv = st.number_input("SLV (Silver)", value=15.0, step=0.1)
        year = st.number_input("Year", min_value=2000, max_value=2100, value=2024)
        month = st.selectbox("Month", list(range(1, 13)))

    st.divider()

    # 6. Prediction Logic
    if st.button("Generate Prediction"):
        # Features array (must match the 6 features the scaler expects)
        features = np.array([[spx, gld, uso, slv, year, month]])
        
        # Transformation and Prediction
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        
        # Clean Results Display (No Balloons)
        st.markdown(f"""
            <div class="prediction-box">
                <p style="color: #6c757d; font-size: 18px; margin-bottom: 5px;">Estimated GLD Value</p>
                <h1 style="color: #212529; font-size: 48px; margin: 0;">${prediction[0]:,.2f}</h1>
            </div>
        """, unsafe_allow_html=True)

# 7. Sidebar Info (Optional - makes it look professional)
with st.sidebar:
    st.header("About the Model")
    st.info("""
    **Algorithm:** K-Nearest Neighbors (KNN)  
    **Scaling:** StandardScaler  
    **Features:** 6 Market Variables  
    """)
    st.caption("Final Project - AI Engineering Track")
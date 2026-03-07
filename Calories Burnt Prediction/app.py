import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="BurnAI | Calories Tracker",
    page_icon="⚡",
    layout="wide"
)

# --- CSS مخصص لجعل الواجهة خرافية ---
st.markdown("""
    <style>
    /* تغيير الخلفية */
    .stApp {
        background-color: #050505;
    }
    /* تنسيق الحاويات */
    [data-testid="stVerticalBlock"] {
        background-color: #111111;
        padding: 20px;
        border-radius: 20px;
        border: 1px solid #222;
    }
    /* زر الحساب */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        color: black;
        font-weight: bold;
        border: none;
        padding: 15px;
        border-radius: 12px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0px 0px 20px #4facfe;
    }
    /* تنسيق الأرقام */
    .big-font {
        font-size: 50px !important;
        font-weight: 800;
        color: #00f2fe;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- تحميل الملفات ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('model.joblib')
        scaler = joblib.load('scaler.joblib')
        return model, scaler
    except:
        return None, None

model, scaler = load_assets()

# ---Header ---
st.markdown("<h1 style='text-align: center; color: white;'>⚡ Burnt Calories AI Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>AI-Powered Calorie Expenditure Analysis</p>", unsafe_allow_html=True)

if not model or not scaler:
    st.warning("⚠️ Files (model.joblib / scaler.joblib) are missing.")
    st.stop()

# --- الجسم الرئيسي للتطبيق ---
main_col1, main_col2 = st.columns([1, 1], gap="large")

with main_col1:
    st.subheader("👤 Personal Profile")
    gender = st.segmented_control("Gender", ["Male", "Female"], default="Male")
    age = st.slider("Age", 10, 90, 25)
    
    col_dim1, col_dim2 = st.columns(2)
    with col_dim1:
        height = st.number_input("Height (cm)", value=170)
    with col_dim2:
        weight = st.number_input("Weight (kg)", value=70)

with main_col2:
    st.subheader("🏃 Training Session")
    duration = st.number_input("Duration (min)", min_value=1, value=30)
    heart_rate = st.slider("Heart Rate (BPM)", 60, 200, 110)
    body_temp = st.select_slider("Body Intensity (Temp)", options=[36, 37, 38, 39, 40, 41, 42], value=37)

# --- معالجة البيانات ---
gender_numeric = 0 if gender == "Male" else 1
input_data = np.array([[gender_numeric, age, height, weight, duration, heart_rate, body_temp]])

st.markdown("<br>", unsafe_allow_html=True)

# --- زر التوقع والنتيجة ---
if st.button("CALCULATE NOW"):
    # 1. Scaling
    scaled_input = scaler.transform(input_data)
    # 2. Prediction
    res = model.predict(scaled_input)[0]
    
    st.markdown("---")
    
    # عرض النتيجة بشكل ضخم ومحترف
    st.markdown('<p style="text-align: center; color: white; font-size: 20px;">Estimated Burned Calories</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="big-font">{round(res, 1)} <span style="font-size: 20px;">KCAL</span></p>', unsafe_allow_html=True)
    
    # إضافة شريط تقدم بصري
    progress_val = min(int(res / 10), 100) # افتراضياً أن 1000 سعرة هي الحد الأقصى للجلسة
    st.progress(progress_val / 100)
    
    # Metrics إضافية
    m1, m2, m3 = st.columns(3)
    m1.metric("Burn Rate", f"{round(res/duration, 1)} kcal/m")
    m2.metric("Heart Load", f"{'High' if heart_rate > 140 else 'Optimal'}")
    m3.metric("Status", "🔥 Shredding" if res > 200 else "Active")
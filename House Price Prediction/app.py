import streamlit as st
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load("house_price_prediction.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="House Price Predictor", layout="wide")

st.title("🏡 California House Price Prediction App")

st.write("Enter housing information to estimate the price.")

col1, col2 = st.columns(2)

with col1:
    longitude = st.number_input("Longitude", value=-122.23)
    latitude = st.number_input("Latitude", value=37.88)
    housing_median_age = st.number_input("Housing Median Age", min_value=1, value=20)
    total_rooms = st.number_input("Total Rooms", min_value=1, value=2000)
    total_bedrooms = st.number_input("Total Bedrooms", min_value=1, value=400)

with col2:
    population = st.number_input("Population", min_value=1, value=1000)
    households = st.number_input("Households", min_value=1, value=300)
    median_income = st.number_input("Median Income", min_value=0.0, value=5.0)

    ocean_proximity = st.selectbox(
        "Ocean Proximity",
        ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
    )

if st.button("Predict Price"):

    # Create dataframe
    input_data = pd.DataFrame({
        "longitude":[longitude],
        "latitude":[latitude],
        "housing_median_age":[housing_median_age],
        "total_rooms":[total_rooms],
        "total_bedrooms":[total_bedrooms],
        "population":[population],
        "households":[households],
        "median_income":[median_income],
        "ocean_proximity":[ocean_proximity]
    })

    # One-hot encode categorical column
    input_data = pd.get_dummies(input_data)

    # Ensure same columns as training
    for col in scaler.feature_names_in_:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[scaler.feature_names_in_]

    # Scale data
    scaled_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_data)

    price = prediction[0]

    st.success(f"🏠 Estimated House Price: ${price:,.2f}")

    st.metric("Predicted Price", f"${price:,.0f}")
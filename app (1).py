import streamlit as st
import pandas as pd
import joblib

# Load saved files
rf = joblib.load("fertilizer_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title("🌱 Fertilizer Recommendation System")

# User inputs
temp = st.number_input("Temperature", value=28.0)
moisture = st.number_input("Moisture", value=0.65)
rainfall = st.number_input("Rainfall", value=120.0)
ph = st.number_input("pH", value=6.5)
nitrogen = st.number_input("Nitrogen", value=60.0)
phosphorous = st.number_input("Phosphorous", value=55.0)
potassium = st.number_input("Potassium", value=50.0)
carbon = st.number_input("Carbon", value=1.5)

soil = st.selectbox("Soil Type", label_encoders["Soil"].classes_)
crop = st.selectbox("Crop Type", label_encoders["Crop"].classes_)

if st.button("Recommend Fertilizer"):
    soil_encoded = label_encoders["Soil"].transform([soil])[0]
    crop_encoded = label_encoders["Crop"].transform([crop])[0]

    input_data = pd.DataFrame([[temp, moisture, rainfall, ph, nitrogen,
                                phosphorous, potassium, carbon,
                                soil_encoded, crop_encoded]],
                              columns=[
                                  "Temperature", "Moisture", "Rainfall", "PH",
                                  "Nitrogen", "Phosphorous", "Potassium",
                                  "Carbon", "Soil", "Crop"
                              ])

    input_scaled = scaler.transform(input_data)
    prediction = rf.predict(input_scaled)

    fertilizer_name = label_encoders["Fertilizer"].inverse_transform(prediction)

    st.success(f"Recommended Fertilizer: {fertilizer_name[0]}")
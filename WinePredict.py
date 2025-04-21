import streamlit as st 
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the model
with open("wine_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit App
st.set_page_config(page_title="Wine Quality Predictor", layout="centered")
st.title("🍷 Wine Quality Prediction App")
st.write("Enter the chemical properties of wine to predict if it's **Good** or **Not Good**.")

# Input fields
fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, step=0.1)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, step=0.01)
citric_acid = st.number_input("Citric Acid", min_value=0.0, step=0.01)
residual_sugar = st.number_input("Residual Sugar", min_value=0.0, step=0.1)
chlorides = st.number_input("Chlorides", min_value=0.0, step=0.0001, format="%.4f")
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, step=1.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, step=1.0)
density = st.number_input("Density", min_value=0.0, step=0.0001, format="%.4f")
pH = st.number_input("pH", min_value=0.0, step=0.01)
sulphates = st.number_input("Sulphates", min_value=0.0, step=0.01)
alcohol = st.number_input("Alcohol", min_value=0.0, step=0.1)

# Prediction
if st.button("Predict Wine Quality"):
    # Input array
    input_data = np.array([[
        fixed_acidity, volatile_acidity, citric_acid,
        residual_sugar, chlorides, free_sulfur_dioxide,
        total_sulfur_dioxide, density, pH, sulphates, alcohol
    ]])

    # Standardize input
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    label = "🟢 Good Quality Wine" if prediction == 1 else "🔴 Not Good Quality Wine"
    st.subheader(f"Prediction: {label}")

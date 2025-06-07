# 📁 app.py - Streamlit Interface for Wine Quality Prediction

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model
@st.cache_resource
def load_model():
    return joblib.load("wine_model.pkl")

model = load_model()

# Streamlit UI
st.title("🍷 Wine Quality Prediction App")

st.header("Input Features")

def user_input_features():
    return pd.DataFrame({
        'fixed acidity': [st.slider('Fixed Acidity', 4.0, 16.0, 7.4)],
        'volatile acidity': [st.slider('Volatile Acidity', 0.10, 1.60, 0.7)],
        'citric acid': [st.slider('Citric Acid', 0.0, 1.0, 0.0)],
        'residual sugar': [st.slider('Residual Sugar', 0.5, 15.0, 1.9)],
        'chlorides': [st.slider('Chlorides', 0.01, 0.20, 0.076)],
        'free sulfur dioxide': [st.slider('Free Sulfur Dioxide', 1, 70, 11)],
        'total sulfur dioxide': [st.slider('Total Sulfur Dioxide', 6, 300, 34)],
        'density': [st.slider('Density', 0.990, 1.005, 0.9978)],
        'pH': [st.slider('pH', 2.8, 4.0, 3.51)],
        'sulphates': [st.slider('Sulphates', 0.2, 2.0, 0.56)],
        'alcohol': [st.slider('Alcohol', 8.0, 15.0, 9.4)],
    })

input_df = user_input_features()

if st.button("Predict Quality"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Wine Quality: {np.round(prediction[0], 2)}")

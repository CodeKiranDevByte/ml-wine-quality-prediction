import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("winequality-red.csv")

df = load_data()

# Title
st.title("🍷 Wine Quality Prediction App")

# Sidebar - Feature input
st.sidebar.header("Input Features")

def user_input_features():
    fixed_acidity = st.sidebar.slider('Fixed Acidity', 4.0, 16.0, 7.4)
    volatile_acidity = st.sidebar.slider('Volatile Acidity', 0.10, 1.60, 0.7)
    citric_acid = st.sidebar.slider('Citric Acid', 0.0, 1.0, 0.0)
    residual_sugar = st.sidebar.slider('Residual Sugar', 0.5, 15.0, 1.9)
    chlorides = st.sidebar.slider('Chlorides', 0.01, 0.20, 0.076)
    free_sulfur_dioxide = st.sidebar.slider('Free Sulfur Dioxide', 1, 70, 11)
    total_sulfur_dioxide = st.sidebar.slider('Total Sulfur Dioxide', 6, 300, 34)
    density = st.sidebar.slider('Density', 0.990, 1.005, 0.9978)
    pH = st.sidebar.slider('pH', 2.8, 4.0, 3.51)
    sulphates = st.sidebar.slider('Sulphates', 0.2, 2.0, 0.56)
    alcohol = st.sidebar.slider('Alcohol', 8.0, 15.0, 9.4)

    data = {
        'fixed acidity': fixed_acidity,
        'volatile acidity': volatile_acidity,
        'citric acid': citric_acid,
        'residual sugar': residual_sugar,
        'chlorides': chlorides,
        'free sulfur dioxide': free_sulfur_dioxide,
        'total sulfur dioxide': total_sulfur_dioxide,
        'density': density,
        'pH': pH,
        'sulphates': sulphates,
        'alcohol': alcohol
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Train model
X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
prediction = model.predict(input_df)

st.subheader("Predicted Wine Quality:")
st.write(np.round(prediction[0], 2))

# Evaluate model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.subheader("Model Evaluation Metrics")
st.write(f"RMSE: {rmse:.2f}")
st.write(f"R² Score: {r2:.2f}")

# Visualizations
st.subheader("Actual vs Predicted Quality")
fig, ax = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred, ax=ax)
ax.set_xlabel("Actual Quality")
ax.set_ylabel("Predicted Quality")
st.pyplot(fig)

import streamlit as st
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("models/heart_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("❤️ Heart Disease Prediction App")

st.write("Enter patient details below:")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Gender", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 90, 200, 120)
chol = st.number_input("Serum Cholesterol (mg/dl)", 120, 350, 180)

# Convert gender
sex_value = 1 if sex == "Male" else 0

if st.button("Predict"):
    input_data = pd.DataFrame(
        [[age, sex_value, cp, trestbps, chol]],
        columns=["age", "sex", "cp", "trestbps", "chol"]
    )

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    st.write(f"### Probability of Heart Disease: {probability*100:.2f}%")

    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")

import joblib
import numpy as np
import pandas as pd

# Load saved model and scaler
model = joblib.load("../models/heart_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

# Feature names (must match training order)
features = [
    ("age", "Age"),
    ("sex", "Sex (1 = Male, 0 = Female)"),
    ("cp", "Chest Pain Type (0-3)"),
    ("trestbps", "Resting Blood Pressure (90-200 mm Hg)"),
    ("chol", "Serum Cholesterol (120-350 mg/dl)"),
]


print("Enter Patient Details:")

user_input = []

for key, label in features:
    value = float(input(f"Enter {label}: "))
    user_input.append(value)


# Convert to numpy array and scale
input_df = pd.DataFrame([user_input], columns=["age", "sex", "cp", "trestbps", "chol"])
input_scaled = scaler.transform(input_df)

# Predict
prediction = model.predict(input_scaled)
probability = model.predict_proba(input_scaled)[0][1]

print(f"\nProbability of Heart Disease: {probability*100:.2f}%")

if prediction[0] == 1:
    print("Prediction: Heart Disease Present")
else:
    print("Prediction: No Heart Disease")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Load data
df = pd.read_csv("../data/heart.csv")

X = df[["age", "sex", "cp", "trestbps", "chol"]]
y = df["target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("===================================")
    print(f"{name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

from sklearn.model_selection import cross_val_score
import numpy as np

print("\n==============================")
print("Cross Validation Results:")
print("==============================")

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name} Mean Accuracy: {np.mean(scores):.4f}")

# Selecting Best Model Manually (Based on Accuracy)

best_model = RandomForestClassifier()
best_model.fit(X_train, y_train)

print("\nBest Model Selected: Random Forest")
print("Final Accuracy:", accuracy_score(y_test, best_model.predict(X_test)))

import joblib

joblib.dump(best_model, "../models/heart_model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")

print("\nModel and Scaler saved successfully!")

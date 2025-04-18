import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Check if dataset exists
if not os.path.exists("data.csv"):
    print("❌ Error: 'data.csv' not found! Place the dataset in the working directory.")
    exit()

print("📂 Loading dataset...")
data = pd.read_csv("data.csv")

# Handle categorical data if any
for col in data.select_dtypes(include=["object"]).columns:
    if col != "status":  # Avoid encoding target variable
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

# Split features and target variable
X = data.drop(columns=["status"])  # Ensure "status" is correct
y = data["status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("🛠️ Training XGBoost model...")

xgb_model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", use_label_encoder=False)

xgb_model.fit(X_train_scaled, y_train)

# Save the model
joblib.dump(xgb_model, "xgb_model.pkl")
joblib.dump(scaler, "scaler_xgb.pkl")
print("✅ XGBoost model saved successfully as 'xgb_model.pkl'")

y_pred = xgb_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Training Complete! Accuracy: {accuracy * 100:.2f}%")
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

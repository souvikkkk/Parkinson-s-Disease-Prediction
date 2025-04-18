import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
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

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure only numerical columns are used
X_train = X_train.select_dtypes(include=['number'])
X_test = X_test.select_dtypes(include=['number'])

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM model
print("🛠️ Training SVM model...")
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')  # Change kernel if needed
svm_model.fit(X_train_scaled, y_train)

# Predictions
y_pred = svm_model.predict(X_test_scaled)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Training Complete! Accuracy: {accuracy * 100:.2f}%")
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# Save the model and scaler
try:
    joblib.dump(svm_model, "svm_model.pkl")
    joblib.dump(scaler, "scaler_svm.pkl")
    print("✅ SVM model saved successfully as 'svm_model.pkl'")
    print("✅ Scaler saved successfully as 'scaler_svm.pkl'")
except Exception as e:
    print(f"❌ Error saving model: {e}")

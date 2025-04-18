import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load Dataset
df = pd.read_csv("data.csv")  # Ensure data.csv is available in the same directory
df.drop(["name"], axis=1, inplace=True)  # Remove 'name' column
df["status"] = df["status"].astype("uint8")  # Convert target column to integer

# Define Features and Target
X = df.drop(["status"], axis=1)
y = df["status"]

# Apply SMOTE to balance the dataset
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

# Normalize features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Save the scaler for future use
joblib.dump(scaler, "scaler_rf.pkl")

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(rf_model, "random_forest.pkl")
print("✅ Random Forest model saved successfully as 'random_forest.pkl'") 

# Save the scaler for feature transformation
joblib.dump(scaler, "scaler_rf.pkl")
print("✅ Scaler saved successfully as 'scaler_rf.pkl'")

# Model Evaluation
y_pred = rf_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))

# train_model.py
import pandas as pd
import numpy as np
import requests
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE # type: ignore
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Step 1: Download and Load Dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data'
data = requests.get(url).content
with open('data.csv', 'wb') as f:
    f.write(data)

df = pd.read_csv('data.csv')
df.drop(['name'], axis=1, inplace=True)  # Remove name column
df['status'] = df['status'].astype('uint8')  # Convert status to integer

# Step 2: Handle Imbalance & Normalize Data
X = df.drop(['status'], axis=1)
y = df['status']

# Apply SMOTE to balance dataset
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

# Normalize features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Step 3: Train & Save Model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

# Train Decision Tree model
model = DecisionTreeClassifier(random_state=42, max_depth=6, criterion='entropy')
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'dt_clf.pkl')

# Step 4: Model Evaluation
y_pred = model.predict(X_test)
print("Model Evaluation:\n", classification_report(y_test, y_pred))
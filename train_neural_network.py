import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score



# Load the dataset
df = pd.read_csv('data.csv')

# Drop 'name' column if it exists (since it's non-numeric)
if 'name' in df.columns:
    df = df.drop(columns=['name'])

# Verify column names
print(df.columns)

# Check if 'status' (target) column exists
if 'status' not in df.columns:
    raise KeyError("Target column 'status' not found. Check column names.")

# Ensure 'status' is numeric
df['status'] = df['status'].astype(int)

# Separate features and target
X = df.drop(columns=["status"]).astype(float)  # Convert features to float
y = df["status"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the model
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Make predictions
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Calculate accuracy manually
accuracy_manual = accuracy_score(y_test, y_pred) * 100
print(f"Manual Accuracy: {accuracy_manual:.2f}%")

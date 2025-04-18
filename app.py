from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import joblib

# Load trained model & scaler
model = joblib.load('dt_clf.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        features = [float(x) for x in request.form.values()]
        input_data = np.array(features).reshape(1, -1)

        # Normalize input data
        scaled_data = scaler.transform(input_data)

        # Predict using trained model
        prediction = model.predict(scaled_data)[0]

        if prediction == 1:
            return redirect(url_for('about'))  # Redirect if Parkinson’s detected
        else:
            return render_template('result.html', prediction="Person does NOT have Parkinson's Disease")
    
    except:
        return render_template('index.html', prediction_text="Error in processing input. Please check the values.")

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
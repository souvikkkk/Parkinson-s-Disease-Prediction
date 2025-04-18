# Parkinson's Disease Prediction using Machine Learning
This project aims to predict whether a person has Parkinson's disease using machine learning algorithms in Python. It utilizes biomedical voice measurements from a dataset to make the prediction.
# 🧠 Project Overview
Parkinson’s disease is a progressive nervous system disorder that affects movement. Early detection can help in effective treatment. In this project, we use machine learning models to classify patients as having Parkinson’s disease or not, based on voice features extracted from sound recordings.
# DataSet
i)   The dataset used is the Parkinson’s dataset from the UCI Machine Learning Repository.    
ii)  Link to dataset    
iii) It contains 24 biomedical voice measurements from 31 people, 23 with Parkinson’s disease.     
# Key Features:
**MDVP:Fo(Hz)** – Average vocal fundamental frequency    
**MDVP:Jitter(%)** – Variation in frequency      
**MDVP:Shimmer** – Variation in amplitude     
**NHR, HNR** – Noise to Harmonic ratio, etc.   
**Status** – Target variable (1 = Parkinson’s, 0 = Healthy)    
# 🛠️Technologies Used:
Python    
Scikit-learn    
Pandas    
NumPy    
Matplotlib / Seaborn (for visualization)     
Flask (for web interface)     
# 📊 Machine Learning Models
Decision Tress      
Support Vector Machine (SVM)     
Random Forest     
Neural Network    
XGBoost     
# 🚀 How to Run
1. Clone the repository:     
   git clone https://github.com/yourusername/parkinsons-prediction.git
cd parkinsons-prediction

2. Install dependencies:     
   pip install -r requirements.txt     

3. Run the model:    
   python train_model.py   
   python random_forest.py   
   python xgboost.py   
   python install svm.py         
   python install neural_network.py        

4. Run the Flask app:     
   python app.py    

# 📈 Results
**Best model:** SVM (or whichever gave best accuracy)    
**Accuracy**: ~90%+    
Confusion matrix, classification report, and ROC curve are used for evaluation

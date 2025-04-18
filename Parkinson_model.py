#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 1Importing necessary libraries
import numpy as np
import pandas as pd
import requests
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score, ConfusionMatrixDisplay


# In[3]:


# 2  URL for data file
url_string = 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data'


# In[4]:


# 3 Downloading content from URL and storing it in a local file
url_content = requests.get(url_string).content
with open('data.csv', 'wb') as data_file:
    data_file.write(url_content)

print("Data downloaded and saved as 'data.csv'")


# In[5]:


# 4 Reading Data Into Pandas DataFrame
df = pd.read_csv('data.csv')


# In[6]:


# 5 Exploring Dataset Content
print("Dataset Head:")
print(df.head())
print("\nDataset Tail:")
print(df.tail())
print('Number of Features In Dataset:', df.shape[1])
print('Number of Instances In Dataset:', df.shape[0])


# In[7]:


# 6 Dropping the Name Column
df.drop(['name'], axis=1, inplace=True)


# In[8]:


# 7
print('Number of Features In Dataset:', df.shape[1])
print('Number of Instances In Dataset:', df.shape[0])


# In[9]:


# 8 Exploring Information About DataFrame
df.info()
print("\nData Description:")
print(df.describe())


# In[10]:


# 9 Converting status to unsigned integer
df['status'] = df['status'].astype('uint8')


# In[11]:


# 10 Checking For Duplicate Rows In Dataset
print('Number of Duplicated Rows:', df.duplicated().sum())


# In[12]:


# 11 Checking For Missing Values In Dataset
print("Missing Values in Dataset:")
print(df.isna().sum())


# In[13]:


# 12 Visualizing the balance of data
sns.countplot(x='status', data=df)
plt.title('Balance of Data')
plt.show()


# In[14]:


# 12 Plotting the correlation heatmap
fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(df.corr(), annot=True, ax=ax)
plt.title('Correlation Heatmap')
plt.show()


# In[15]:


# Box Plot for features
fig, axes = plt.subplots(5, 5, figsize=(15, 15))
axes = axes.flatten()
for i in range(1, len(df.columns)-1):
    sns.boxplot(x='status', y=df.iloc[:, i], data=df, orient='v', ax=axes[i])
plt.tight_layout()
plt.show()


# In[16]:


# Pair plots for selected features
plt.rcParams['figure.figsize'] = (15, 4)
sns.pairplot(df, hue='status', vars=['MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP'])
plt.show()


# In[17]:


plt.rcParams['figure.figsize'] = (15, 4)
sns.pairplot(df, hue='status', vars=['MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA'])
plt.show()


# In[18]:


# Exploring Imbalance in Dataset
print("Status Value Counts:")
print(df['status'].value_counts())


# In[19]:


# Extracting Features and Target
X = df.drop(['status'], axis=1)
y = df['status']
print('Feature (X) Shape Before Balancing:', X.shape)
print('Target (y) Shape Before Balancing:', y.shape)


# In[20]:


# Initializing SMOTE Object
sm = SMOTE(random_state=300)
# Resampling Data
X, y = sm.fit_resample(X, y)
print('Feature (X) Shape After Balancing:', X.shape)
print('Target (y) Shape After Balancing:', y.shape)


# In[21]:


# Scaling features between -1 and 1 for normalization
scaler = MinMaxScaler((-1, 1))
X_features = scaler.fit_transform(X)
Y_labels = y


# In[22]:


# Splitting the dataset into training and testing sets (80-20)
X_train, X_test, y_train, y_test = train_test_split(X_features, Y_labels, test_size=0.20, random_state=20)


# In[23]:


# Training Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
predDT = clf.predict(X_test)


# In[24]:


# Printing the classification report
print(classification_report(y_test, predDT))


# In[25]:


# Tuning Hyperparameters using GridSearchCV
param_grid = {
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': range(1, 10),
    'random_state': range(30, 210, 30),
    'criterion': ['gini', 'entropy']
}
CV_dt = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)
CV_dt.fit(X_train, y_train)


# In[26]:


# Best parameters from GridSearchCV
print("Best Parameters:", CV_dt.best_params_)


# In[27]:


# Printing the classification report
print(classification_report(y_test, predDT))


# In[28]:


# Initialize Decision Tree Classifier with valid parameters
dt1 = DecisionTreeClassifier(random_state=120, max_features='sqrt', max_depth=6, criterion='entropy')


# In[29]:


# Fit the model
dt1.fit(X_train, y_train)


# In[30]:


# Make predictions
predDT = dt1.predict(X_test)


# In[31]:


# Print the classification report
print(classification_report(y_test, predDT))


# In[32]:


# Compute the confusion matrix
cm = confusion_matrix(y_test, predDT)

# Plotting the confusion matrix using seaborn heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Parkinsons', 'Parkinsons'], yticklabels=['No Parkinsons', 'Parkinsons'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix for Decision Tree')
plt.show()


# In[33]:


# Getting predicted probabilities
y_pred_proba = dt1.predict_proba(X_test)[:, 1]

# Calculating false positive rate and true positive rate
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

# Plotting the ROC curve
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, label="AUC = {:.2f}".format(auc))
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[34]:


# Dumping Decision Tree Classifier
joblib.dump(dt1, 'dt_clf.pkl')


# In[35]:


from sklearn.preprocessing import StandardScaler
import joblib

# Assume `scaler` is your fitted StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)  # X_train is your training data

# Save the fitted scaler to a file
joblib.dump(scaler, 'scaler.pkl')


# In[41]:


#0=NO 0
#1=YES 1

import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('dt_clf.pkl')  # Load your trained model
scaler = joblib.load('scaler.pkl')  # Load your fitted scaler

# Input data for prediction
input_values = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)

# Convert input data to a numpy array
input_array = np.asarray(input_values)

# Reshape the numpy array
reshaped_input = input_array.reshape(1, -1)

# Standardize the data
standardized_data = scaler.transform(reshaped_input)

# Make the prediction
output_prediction = model.predict(standardized_data)

# Display the prediction result
if output_prediction[0] == 0:
    print("The Person does not have Parkinson's Disease")
else:
    print("The Person has Parkinson's Disease")


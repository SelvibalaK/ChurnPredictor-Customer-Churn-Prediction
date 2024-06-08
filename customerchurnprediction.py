# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 23:30:44 2024

@author: Selvibala
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import random

# Loading the dataset 
data = pd.read_csv(r'C:\Users\Selvibala\Downloads\customerchurnprediction\Churn_Modelling.csv')

# Selecting the relevant features
selected_features = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]

X = data[selected_features]
y = data["Exited"]  # Target variable (1 if customer churned, 0 otherwise)

# Encoding categorical features
X = pd.get_dummies(X, columns=["Geography", "Gender"], drop_first=True)

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training with the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Randomly select a customer churn record
random_index = random.randint(0, len(X_test) - 1)
selected_record = X_test.iloc[random_index]
print(f"Selected Customer Record:\n{selected_record}\n")

# Transform the selected record
X_selected_scaled = scaler.transform([selected_record])

# Make predictions
predicted_churn = model.predict(X_selected_scaled)[0]
if predicted_churn == 1:
    print("Predicted Churn: Customer is likely to churn")
else:
    print("Predicted Churn: Customer is unlikely to churn")

# Calculate and display model accuracy
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

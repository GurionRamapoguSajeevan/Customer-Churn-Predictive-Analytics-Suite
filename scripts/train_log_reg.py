"""
Train a Baseline Logistic Regression Model
-------------------------------------------
Uses SMOTE-balanced training data to fit a logistic regression classifier.
Evaluates on the untouched test set using confusion matrix, classification report, and ROC-AUC score.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib

# -----------------------------
# Load split & resampled datasets
# -----------------------------
X_train_res = pd.read_csv("data/X_train_smote.csv")
y_train_res = pd.read_csv("data/y_train_smote.csv").squeeze()  # convert to Series
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Train Logistic Regression model
# -----------------------------
lr = LogisticRegression(max_iter=5000, random_state=42)
lr.fit(X_train_res_scaled, y_train_res)

# -----------------------------
# Predictions & Probabilities
# -----------------------------
y_pred = lr.predict(X_test_scaled)
y_proba = lr.predict_proba(X_test_scaled)[:, 1]

# -----------------------------
# Evaluation
# -----------------------------
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

print("\nROC-AUC Score:", roc_auc_score(y_test, y_proba))

# -----------------------------
# Save model & scaler
# -----------------------------
import os
import numpy as np

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

joblib.dump(lr, "models/logistic_regression.pkl")
joblib.dump(scaler, "models/scaler.pkl")

pd.DataFrame(X_train_res_scaled, columns=X_train_res.columns).to_csv("data/X_train_scaled.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv("data/X_test_scaled.csv", index=False)

print("\nModel, scaler, and scaled datasets saved successfully.")
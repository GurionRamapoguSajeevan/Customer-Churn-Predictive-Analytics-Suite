"""
Ensemble Model: Voting Classifier for Customer Churn Prediction
---------------------------------------------------------------
Combines Logistic Regression, tuned Random Forest, and tuned XGBoost
using soft voting to improve predictive performance.
"""

import pandas as pd
import joblib
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# -----------------------------
# Load train/test data
# -----------------------------
X_train_scaled = pd.read_csv("data/X_train_scaled.csv")  # Scaled SMOTE train set
y_train_res = pd.read_csv("data/y_train_smote.csv").squeeze()  # <-- fixed here
X_test_scaled = pd.read_csv("data/X_test_scaled.csv")    # Scaled test set
y_test = pd.read_csv("data/y_test.csv").squeeze()        # <-- fixed here

# -----------------------------
# Load trained base models
# -----------------------------
lr = joblib.load("models/logistic_regression.pkl")
best_rf = joblib.load("models/random_forest_best.pkl")
best_xgb = joblib.load("models/xgboost_best.pkl")

# -----------------------------
# Create soft voting ensemble
# -----------------------------
ensemble = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('rf', best_rf),
        ('xgb', best_xgb)
    ],
    voting='soft'
)

# Train the ensemble
ensemble.fit(X_train_scaled, y_train_res)

# -----------------------------
# Predictions & Evaluation
# -----------------------------
y_pred_ensemble = ensemble.predict(X_test_scaled)
y_proba_ensemble = ensemble.predict_proba(X_test_scaled)[:, 1]

print("\nEnsemble Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_ensemble))

print("\nEnsemble Classification Report:")
print(classification_report(y_test, y_pred_ensemble, digits=4))

print("\nEnsemble ROC-AUC Score:", roc_auc_score(y_test, y_proba_ensemble))

# -----------------------------
# Save ensemble model & outputs
# -----------------------------
import os
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

joblib.dump(ensemble, "models/ensemble_voting.pkl")

# Optional: Save predictions/probabilities for later visualization
pd.Series(y_pred_ensemble).to_csv("data/y_pred_ensemble.csv", index=False)
pd.Series(y_proba_ensemble).to_csv("data/y_proba_ensemble.csv", index=False)

print("\n✅ Ensemble model and outputs saved.")

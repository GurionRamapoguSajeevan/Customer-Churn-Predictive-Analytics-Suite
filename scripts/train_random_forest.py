"""
Train and Tune Random Forest Classifier with SMOTE-balanced Data
----------------------------------------------------------------
Includes:
- Scaling features
- Training baseline Random Forest
- Hyperparameter tuning using RandomizedSearchCV
- Evaluating performance on test set
- Plotting top 15 feature importances
- Generating SHAP explanations and summary plot for churn prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import shap

# -----------------------------
# Load train/test data
# -----------------------------
X_train_res = pd.read_csv("data/X_train_smote.csv")
y_train_res = pd.read_csv("data/y_train_smote.csv").squeeze()  # Series needed for sklearn
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()

# -----------------------------
# Scale features
# -----------------------------
scaler = StandardScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Train baseline Random Forest
# -----------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_res_scaled, y_train_res)

# Predict and evaluate baseline
y_pred_rf = rf.predict(X_test_scaled)
y_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]

print("Baseline Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print("\nBaseline Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf, digits=4))
print("\nBaseline Random Forest ROC-AUC Score:", roc_auc_score(y_test, y_proba_rf))

# -----------------------------
# Hyperparameter tuning with RandomizedSearchCV
# -----------------------------
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf_tune = RandomForestClassifier(random_state=42)

rand_search = RandomizedSearchCV(
    estimator=rf_tune,
    param_distributions=param_dist,
    n_iter=20,
    cv=3,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1,
    verbose=2
)

rand_search.fit(X_train_res_scaled, y_train_res)

best_rf = rand_search.best_estimator_
print("\nBest parameters found:", rand_search.best_params_)

# Predict and evaluate tuned model
y_pred_best_rf = best_rf.predict(X_test_scaled)
y_proba_best_rf = best_rf.predict_proba(X_test_scaled)[:, 1]

print("Tuned Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_best_rf))
print("\nTuned Random Forest Classification Report:")
print(classification_report(y_test, y_pred_best_rf, digits=4))
print("\nTuned Random Forest ROC-AUC Score:", roc_auc_score(y_test, y_proba_best_rf))

# -----------------------------
# Feature importance plot (top 15)
# -----------------------------

# Convert scaled training data back to DataFrame for feature names
X_train_df = pd.DataFrame(X_train_res_scaled, columns=X_train_res.columns)

feature_imp = pd.Series(best_rf.feature_importances_, index=X_train_df.columns).sort_values(ascending=False)

top_features = feature_imp.head(15)

plt.figure(figsize=(10, 6))
colors = sns.color_palette('viridis', n_colors=len(top_features))
plt.barh(top_features.index[::-1], top_features.values[::-1], color=colors[::-1])  # reverse for top-down order
plt.title('Top 15 Feature Importances')
plt.xlabel('Importance')
plt.tight_layout()
plt.gca().invert_yaxis()  # most important on top
plt.show()

# -----------------------------
# SHAP explanation & summary plot
# -----------------------------

# Prepare SHAP explainer
explainer = shap.TreeExplainer(best_rf)

# Use subset of test data for SHAP explanations (reduce size if very large)
X_test_sample = X_test_scaled[:100]

# Get SHAP values (Explanation object)
shap_values_exp = explainer(X_test_sample)

print("SHAP explanation values shape:", shap_values_exp.values.shape)

# Plot global summary for positive class (class 1 = churn)
shap.summary_plot(shap_values_exp.values[:, :, 1], pd.DataFrame(X_test_sample, columns=X_train_res.columns))

import os
import joblib

os.makedirs("models", exist_ok=True)  # Ensure directory exists

# Save the tuned Random Forest model
joblib.dump(best_rf, "models/random_forest_best.pkl")
print("Tuned Random Forest saved to models/random_forest_best.pkl")

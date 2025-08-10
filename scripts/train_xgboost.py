"""
Train and Tune XGBoost Classifier with SMOTE-balanced Data
----------------------------------------------------------
Includes:
- Baseline model training
- Model evaluation on test set
- Hyperparameter tuning with RandomizedSearchCV
- Feature importance visualization for baseline and tuned models
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

# -----------------------------
# Load train/test data
# -----------------------------
X_train_res = pd.read_csv("data/X_train_smote.csv")
y_train_res = pd.read_csv("data/y_train_smote.csv").squeeze()
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()

# -----------------------------
# Instantiate baseline XGBoost classifier
# -----------------------------
xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

# -----------------------------
# Train baseline model
# -----------------------------
xgb_clf.fit(X_train_res, y_train_res)

# Predict and evaluate baseline model
y_pred_xgb = xgb_clf.predict(X_test)
y_proba_xgb = xgb_clf.predict_proba(X_test)[:, 1]

print("Baseline XGBoost Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))

print("\nBaseline XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb, digits=4))

print("\nBaseline XGBoost ROC-AUC Score:", roc_auc_score(y_test, y_proba_xgb))

# -----------------------------
# Hyperparameter tuning (RandomizedSearchCV)
# -----------------------------
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'reg_lambda': [1, 1.5, 2],
}

random_search = RandomizedSearchCV(
    estimator=xgb_clf,
    param_distributions=param_dist,
    n_iter=20,
    cv=3,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1,
    verbose=2
)

random_search.fit(X_train_res, y_train_res)

best_xgb = random_search.best_estimator_
print("\nBest XGBoost Parameters:", random_search.best_params_)

# Evaluate tuned model
y_pred_best_xgb = best_xgb.predict(X_test)
y_proba_best_xgb = best_xgb.predict_proba(X_test)[:, 1]

print("Tuned XGBoost Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_best_xgb))

print("\nTuned XGBoost Classification Report:")
print(classification_report(y_test, y_pred_best_xgb, digits=4))

print("\nTuned XGBoost ROC-AUC Score:", roc_auc_score(y_test, y_proba_best_xgb))

# -----------------------------
# Feature importance plotting function
# -----------------------------
def plot_xgb_feature_importance(model, feature_names, title, top_n=15):
    # Extract importance scores
    importance_dict = model.get_booster().get_score(importance_type='weight')
    
    imp_df = pd.DataFrame({
        'Feature': [feature_names[int(k[1:])] if k[0] == 'f' else k for k in importance_dict.keys()],
        'Importance': list(importance_dict.values())
    })
    
    imp_df = imp_df.sort_values(by='Importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette('viridis', n_colors=top_n)
    sns.barplot(x='Importance', y='Feature', data=imp_df, palette=colors)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Plot feature importance for baseline and tuned model
plot_xgb_feature_importance(xgb_clf, X_train_res.columns, 'Baseline XGBoost Top 15 Feature Importances')
plot_xgb_feature_importance(best_xgb, X_train_res.columns, 'Tuned XGBoost Top 15 Feature Importances')

import os
import joblib

os.makedirs("models", exist_ok=True)  # Make sure models directory exists
joblib.dump(best_xgb, "models/xgboost_best.pkl")
print("Tuned XGBoost saved to models/xgboost_best.pkl")
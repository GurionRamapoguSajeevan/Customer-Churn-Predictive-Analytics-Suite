"""
Streamlit Dashboard for Telco Customer Churn Prediction
-------------------------------------------------------
Displays ensemble model evaluation metrics, confusion matrix,
ROC and Precision-Recall curves, feature importance plots,
and allows interactive single-customer churn probability prediction.
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve, 
    precision_recall_curve, average_precision_score, precision_score, recall_score
)
import plotly.graph_objects as go
import plotly.express as px

# --- Cache loading function for faster reruns ---
@st.cache_data
def load_data(file_path='model_outputs.pkl'):
    return joblib.load(file_path)

# Load all data/models/scaler from the saved pickle file
data = load_data()

y_test = data['y_test']
y_pred_ensemble = data['y_pred_ensemble']
y_proba_ensemble = data['y_proba_ensemble']
feature_names = data['feature_names']
best_rf = data['best_rf']
best_xgb = data['best_xgb']
scaler = data['scaler']  # assume StandardScaler or similar

# Uncomment if logistic regression model is saved and needed
# best_lr = data.get('best_lr', None)

# Page config and title
st.set_page_config(page_title="Telco Customer Churn Prediction Dashboard", layout="wide")
st.title("Telco Customer Churn Prediction Dashboard")

st.markdown("""
### Project Overview
This dashboard presents evaluation results and insights from ensemble churn prediction models built on the Telco Customer Churn dataset.

Explore performance metrics, curves, feature importance, and try predicting churn for a hypothetical customer interactively.

---
""")

# Display classification report
st.header("Ensemble Model Evaluation Metrics")
report = classification_report(y_test, y_pred_ensemble, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)

# Confusion matrix heatmap
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred_ensemble)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Confusion Matrix")
st.pyplot(fig)

# ROC Curve
st.subheader("ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_proba_ensemble)
roc_auc = roc_auc_score(y_test, y_proba_ensemble)

fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve',
                            line=dict(color='darkorange', width=3)))
fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                            line=dict(color='navy', width=2, dash='dash'),
                            showlegend=False))
fig_roc.update_layout(title=f'ROC Curve (AUC = {roc_auc:.3f})',
                      xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate',
                      width=700, height=500)
st.plotly_chart(fig_roc, use_container_width=True)

# Precision-Recall Curve
st.subheader("Precision-Recall Curve")
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba_ensemble)
avg_precision = average_precision_score(y_test, y_proba_ensemble)

fig_pr = go.Figure()
fig_pr.add_trace(go.Scatter(x=recall_vals, y=precision_vals, mode='lines', name='Precision-Recall Curve',
                            line=dict(color='purple', width=3)))
fig_pr.update_layout(title=f'Precision-Recall Curve (Average Precision = {avg_precision:.3f})',
                     xaxis_title='Recall',
                     yaxis_title='Precision',
                     width=700, height=500)
st.plotly_chart(fig_pr, use_container_width=True)

# Feature Importance - Random Forest
st.header("Top 15 Feature Importances - Random Forest")
feature_imp_rf = pd.Series(best_rf.feature_importances_, index=feature_names)
top15_rf = feature_imp_rf.sort_values(ascending=False).head(15).reset_index()
top15_rf.columns = ['Feature', 'Importance']

fig_rf = px.bar(top15_rf, x='Importance', y='Feature', orientation='h',
                color='Importance', color_continuous_scale='viridis',
                title='Random Forest Feature Importances (Top 15)')
fig_rf.update_layout(yaxis={'categoryorder': 'total ascending'}, width=700, height=500)
st.plotly_chart(fig_rf, use_container_width=True)

# Feature Importance - XGBoost
st.header("Top 15 Feature Importances - XGBoost")
importance_dict = best_xgb.get_booster().get_score(importance_type='weight')
importance_df = pd.DataFrame({
    'Feature': [feature_names[int(k[1:])] if k.startswith('f') else k for k in importance_dict.keys()],
    'Importance': list(importance_dict.values())
})
top15_xgb = importance_df.sort_values(by='Importance', ascending=False).head(15)

fig_xgb = px.bar(top15_xgb, x='Importance', y='Feature', orientation='h',
                 color='Importance', color_continuous_scale='viridis',
                 title='XGBoost Feature Importances (Top 15)')
fig_xgb.update_layout(yaxis={'categoryorder': 'total ascending'}, width=700, height=500)
st.plotly_chart(fig_xgb, use_container_width=True)

# --- Interactive Single Customer Prediction ---
st.header("Predict Churn for a Single Customer")

st.markdown("""
Use the slider below to adjust the classification threshold for predicted churn probability.

Enter feature values for the customer you want to predict.
""")
threshold = st.slider("Set classification threshold", 0.0, 1.0, 0.5, 0.01)

# Example: Define categorical features if any; update as per your feature set
categorical_features = []  # e.g., ['gender', 'ContractType']

with st.form("customer_form"):
    inputs = {}
    for feat in feature_names:
        if feat in categorical_features:
            # Replace with actual categorical options
            options = ['Option 1', 'Option 2', 'Option 3']
            inputs[feat] = st.selectbox(f"{feat} (categorical)", options)
        else:
            inputs[feat] = st.number_input(feat, value=0.0, format="%.4f")

    submitted = st.form_submit_button("Predict Churn")
    if submitted:
        input_df = pd.DataFrame([inputs])

        # TODO: Add categorical encoding here if needed

        # Scale input features
        input_scaled = scaler.transform(input_df)

        # Get predicted probabilities from models
        proba_rf = best_rf.predict_proba(input_scaled)[0, 1]
        proba_xgb = best_xgb.predict_proba(input_scaled)[0, 1]

        # Example: average predictions; include logistic regression if available
        # if best_lr is not None:
        #     proba_lr = best_lr.predict_proba(input_scaled)[0, 1]
        #     proba_ensemble = (proba_rf + proba_xgb + proba_lr) / 3
        # else:
        proba_ensemble = (proba_rf + proba_xgb) / 2

        st.write(f"Predicted probability of churn: **{proba_ensemble:.2%}**")
        if proba_ensemble >= threshold:
            st.warning("This customer is likely to churn.")
        else:
            st.success("This customer is unlikely to churn.")

"""
Interactive Visualizations for Customer Churn Models
----------------------------------------------------
Generates interactive ROC and Precision-Recall curves for the ensemble model,
and interactive feature importance bar charts for tuned Random Forest and XGBoost models using Plotly.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
import joblib

# -----------------------------
# Load predictions and test labels
# -----------------------------
# Load the test set and ensemble prediction probabilities
y_test = pd.read_csv("data/y_test.csv").squeeze()
y_proba_ensemble = pd.read_csv("data/y_proba_ensemble.csv").squeeze()   # Make sure to save this after ensemble prediction step

# Load feature data and tuned models
X = pd.read_csv("data/X_test.csv")  # or whole feature set for importance plots

best_rf = joblib.load("models/random_forest_best.pkl")
best_xgb = joblib.load("models/xgboost_best.pkl")

# -----------------------------
# Interactive ROC Curve for Ensemble Model
# -----------------------------
fpr, tpr, _ = roc_curve(y_test, y_proba_ensemble)

fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve',
                            line=dict(color='darkorange', width=4)))
fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                            line=dict(color='navy', width=2, dash='dash'),
                            showlegend=False))
fig_roc.update_layout(title='Interactive ROC Curve - Ensemble Model',
                      xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate',
                      width=700, height=500)
fig_roc.show()

# -----------------------------
# Interactive Precision-Recall Curve for Ensemble Model
# -----------------------------
precision, recall, _ = precision_recall_curve(y_test, y_proba_ensemble)
avg_precision = average_precision_score(y_test, y_proba_ensemble)

fig_pr = go.Figure()
fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='Precision-Recall Curve',
                            line=dict(color='purple', width=4)))
fig_pr.update_layout(title=f'Interactive Precision-Recall Curve - Ensemble Model (AP={avg_precision:.3f})',
                     xaxis_title='Recall',
                     yaxis_title='Precision',
                     width=700, height=500)
fig_pr.show()

# -----------------------------
# Interactive Feature Importance Bar Chart - Tuned Random Forest
# -----------------------------
feature_imp_rf = pd.Series(best_rf.feature_importances_, index=X.columns)
top15_rf = feature_imp_rf.sort_values(ascending=False).head(15).reset_index()
top15_rf.columns = ['Feature', 'Importance']

fig_rf = px.bar(top15_rf, x='Importance', y='Feature', orientation='h',
               color='Importance', color_continuous_scale='viridis',
               title='Top 15 Feature Importances - Tuned Random Forest')
fig_rf.update_layout(yaxis={'categoryorder':'total ascending'}, width=700, height=500)
fig_rf.show()

# -----------------------------
# Interactive Feature Importance Bar Chart - Tuned XGBoost
# -----------------------------
importance_dict = best_xgb.get_booster().get_score(importance_type='weight')
importance_df = pd.DataFrame({
    'Feature': [X.columns[int(k[1:])] if k.startswith('f') else k for k in importance_dict.keys()],
    'Importance': list(importance_dict.values())
})
top15_xgb = importance_df.sort_values(by='Importance', ascending=False).head(15)

fig_xgb = px.bar(top15_xgb, x='Importance', y='Feature', orientation='h',
                 color='Importance', color_continuous_scale='viridis',
                 title='Top 15 Feature Importances - Tuned XGBoost')
fig_xgb.update_layout(yaxis={'categoryorder':'total ascending'}, width=700, height=500)
fig_xgb.show()
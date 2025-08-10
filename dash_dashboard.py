"""
Dash Dashboard for Customer Churn Prediction Model
--------------------------------------------------
Displays ensemble model performance metrics, confusion matrix,
ROC curve, precision-recall curve, and feature importance charts for
tuned Random Forest and XGBoost models using Plotly and Dash.
"""

import dash
from dash import html, dcc
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
from sklearn.metrics import (
    roc_curve, precision_recall_curve, average_precision_score,
    roc_auc_score, precision_score, recall_score, accuracy_score, confusion_matrix
)
import joblib

# -----------------------------------------------------
# Load required data and models saved from previous scripts
# -----------------------------------------------------

# Features DataFrame (for feature names)
X = pd.read_csv("data/X_test.csv")

# True labels and ensemble predictions saved after ensemble script run
y_test = pd.read_csv("data/y_test.csv").squeeze()
y_pred_ensemble = pd.read_csv("data/y_pred_ensemble.csv").squeeze()
y_proba_ensemble = pd.read_csv("data/y_proba_ensemble.csv").squeeze()

# Load trained models
best_rf = joblib.load("models/random_forest_best.pkl")
best_xgb = joblib.load("models/xgboost_best.pkl")

# -----------------------------------------------------
# Calculate key metrics
# -----------------------------------------------------

accuracy = accuracy_score(y_test, y_pred_ensemble)
precision = precision_score(y_test, y_pred_ensemble)
recall = recall_score(y_test, y_pred_ensemble)
rocauc = roc_auc_score(y_test, y_proba_ensemble)

# -----------------------------------------------------
# Confusion matrix figure
# -----------------------------------------------------

cm = confusion_matrix(y_test, y_pred_ensemble)
cm_fig = ff.create_annotated_heatmap(
    z=cm,
    x=['Pred 0', 'Pred 1'],
    y=['True 0', 'True 1'],
    annotation_text=cm.astype(str),
    colorscale='Blues',
    showscale=False
)
cm_fig.update_layout(title='Confusion Matrix')

# -----------------------------------------------------
# ROC Curve figure
# -----------------------------------------------------

fpr, tpr, _ = roc_curve(y_test, y_proba_ensemble)
roc_fig = go.Figure()
roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve',
                            line=dict(color='darkorange', width=4)))
roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                            line=dict(color='navy', width=2, dash='dash'), showlegend=False))
roc_fig.update_layout(
    title='ROC Curve - Ensemble Model',
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate'
)

# -----------------------------------------------------
# Precision-Recall Curve figure
# -----------------------------------------------------

precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba_ensemble)
avg_precision = average_precision_score(y_test, y_proba_ensemble)
pr_fig = go.Figure()
pr_fig.add_trace(go.Scatter(x=recall_vals, y=precision_vals, mode='lines', name='Precision-Recall Curve',
                            line=dict(color='purple', width=4)))
pr_fig.update_layout(
    title=f'Precision-Recall Curve - Ensemble Model (AP={avg_precision:.3f})',
    xaxis_title='Recall',
    yaxis_title='Precision'
)

# -----------------------------------------------------
# Feature Importance Figures
# -----------------------------------------------------

# Random Forest Feature Importance
feature_imp_rf = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(15).reset_index()
feature_imp_rf.columns = ['Feature', 'Importance']
rf_fig = px.bar(
    feature_imp_rf, x='Importance', y='Feature', orientation='h',
    color='Importance', color_continuous_scale='viridis',
    title='Top 15 Feature Importances - Tuned Random Forest')
rf_fig.update_layout(yaxis={'categoryorder':'total ascending'})

# XGBoost Feature Importance
importance_dict = best_xgb.get_booster().get_score(importance_type='weight')
importance_df = pd.DataFrame({
    'Feature': [X.columns[int(k[1:])] if k.startswith('f') else k for k in importance_dict.keys()],
    'Importance': list(importance_dict.values())
})
top15_xgb = importance_df.sort_values(by='Importance', ascending=False).head(15)
xgb_fig = px.bar(
    top15_xgb, x='Importance', y='Feature', orientation='h',
    color='Importance', color_continuous_scale='viridis',
    title='Top 15 Feature Importances - Tuned XGBoost')
xgb_fig.update_layout(yaxis={'categoryorder':'total ascending'})

# -----------------------------------------------------
# Dash app and layout
# -----------------------------------------------------

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Churn Prediction Model Dashboard', style={'textAlign': 'center'}),
    
    html.Div([
        html.H2('Model Performance Metrics'),
        html.P(f"Ensemble Accuracy: {accuracy:.3f}"),
        html.P(f"Ensemble Precision (Churn): {precision:.3f}"),
        html.P(f"Ensemble Recall (Churn): {recall:.3f}"),
        html.P(f"Ensemble ROC-AUC Score: {rocauc:.3f}"),
        html.Hr(),
        
        html.H3('Confusion Matrix'),
        dcc.Graph(figure=cm_fig),
        
        html.H3('ROC Curve'),
        dcc.Graph(figure=roc_fig),
        
        html.H3('Precision-Recall Curve'),
        dcc.Graph(figure=pr_fig),
        
        html.H3('Top Feature Importances (Random Forest)'),
        dcc.Graph(figure=rf_fig),
        
        html.H3('Top Feature Importances (XGBoost)'),
        dcc.Graph(figure=xgb_fig),
    ], style={'width': '80%', 'margin': 'auto'}),
])

if __name__ == '__main__':
    app.run(debug=True)


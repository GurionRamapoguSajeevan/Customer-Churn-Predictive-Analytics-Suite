# ⭐ Customer-Churn-Predictive-Analytics-Suite
Unlock actionable insights and prediction power with end-to-end churn analytics, combining advanced EDA, explainability, dashboards, and state-of-the-art machine learning.

# Project Overview
The Customer Churn Predictive Analytics Suite is a full-stack analytics and machine learning project designed to analyze customer behavior, uncover churn drivers, and build predictive models that help telecom companies reduce churn.

# The SOLUTION includes:
* Deep exploratory data analysis (EDA)
* Interactive dashboards (Streamlit & Dash)
* Advanced machine learning models
* SHAP-based explainability
* Ensemble modeling for high-performance churn prediction
* Executive insights and business recommendations

# Key Features

* Comprehensive EDA revealing behavioral and financial churn patterns
* Advanced ML modeling : Logistic Regression, Tuned Random Forest, Tuned XGBoost, and soft-voting Ensemble
* Model explainability using SHAP for transparent customer-level predictions
* Interactive dashboards for real-time churn scoring and insights
* Business-driven churn reduction recommendations
* Production-ready pipeline structure suitable for real deployments

# Tech Stack
**Python**: _`Pandas, NumPy, Scikit-learn, SHAP explainer, Logistic Regression, Random Forest, XGBoost, Ensembling`_

**Visualization**: _`Matplotlib, Seaborn, Plotly`_

**Dashboarding**: _`Streamlit, Dash`_

**Environment**: _`Python 3.8+, Jupyter notebook`_

**Version Control**: _`Git & GitHub`_

# Machine Learning & Modeling Pipeline

* Data loading & preprocessing
* Feature engineering (categorical encoding, charge engineering)
* Train-test split with stratification
* SMOTE for class imbalance

Model training:
* Logistic Regression
* Baseline Random Forest
* Random Forest (tuned)
* Baseline XGBoost
* XGBoost (tuned)
* Soft-voting Ensemble (Tuned RF + Tuned XGBoost)

Model performance evaluation:
* Classification reports
* Confusion matrices
* ROC-AUC curves
* Precision–Recall curves

Explainability:
* SHAP bar + beeswarm plots
* Feature importance visualizations

Deployment dashboards:
* Streamlit
* Dash

# 📊 Model Performance Summary
| Model                      | Accuracy  | ROC-AUC    | Churn Recall | Churn Precision |
| -------------------------- | --------- | ---------- | ------------ | --------------- |
| Logistic Regression        | 78.6%     | **0.8367** | **0.6123**   | 0.5948          |
| Tuned Random Forest        | 76.7%     | 0.8230     | 0.6043       | 0.5567          |
| Tuned XGBoost              | 77.0%     | 0.8158     | 0.5642       | 0.5672          |
| **Ensemble (Final Model)** | **78.1%** | **0.8336** | 0.5909       | 0.5878          |

# Explainability: Key Drivers of Churn

Using SHAP analysis from both Random Forest and XGBoost models, the _**Top Predictors**_ are:
* PaymentMethod_Electronic check → High churn risk
* InternetService_Fiber optic → Higher churn sensitivity
* Tenure → Lower tenure = significantly higher churn
* TotalServices → Customers with fewer services churn more
* MonthlyCharges / AvgMonthlyCharge → Pricing dissatisfaction
* Contract_Two year → Long-term contracts reduce churn

Explainability ensures the model is transparent, audit-friendly, and deployable in real-world business settings.

# 📈 Dashboards

1. Streamlit Dashboard : View the dedicated APP through the _Github repo_ here : [_Streamlit_app_Customer_churn_prediction_](https://github.com/GurionRamapoguSajeevan/Streamlit_app_Customer_churn_prediction)
* User-friendly interface
* Customer-level predictions
* SHAP explanations displayed interactively
* Real-time churn scoring and filtering

2. Dash Dashboard
* Analytical layout for deeper data exploration
* Visualizations of churn patterns
* Model metrics and performance charts

# Getting Started
_Prerequisites_
`Python 3.8+`

Packages listed in `requirements.txt`

# Installation

# Clone the repository:
`git clone https://github.com/GurionRamapoguSajeevan/customer-churn-analytics-suite.git`

# Navigate to the project directory:
`cd customer-churn-analytics-suite`

# Install dependencies:
`pip install -r requirements.txt`

# Usage
* Run the EDA notebook to explore data insights: `Telco_Churn_EDA.ipynb`

* Launch the Streamlit dashboard:
`streamlit run streamlit_dashboard.py`

* Launch the Dash dashboard:
`python dash_dashboard.py`

* Train and evaluate models with the provided scripts

# Project Structure

/Customer-Churn-Analytics-Suite

├── data/                      # Dataset files (excluded from Git tracking)

├── notebooks/                 # Jupyter notebooks for EDA and experiments

├── models/                    # Saved model files and outputs

├── dashboards/                # Streamlit and Dash app code

├── scripts/                   # Training, evaluation scripts for ML models

├── requirements.txt           # Python packages required

├── README.md                  # Project overview and setup

├── .gitignore                 # Ignore unnecessary files

└── LICENSE                   # License information

# License
This project is licensed under the _`MIT License`_ - see the LICENSE file for details.

# Contact
For questions or collaborations, reach out to me at _`GurionRamapoguSajeevan`_.

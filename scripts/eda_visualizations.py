"""
Basic Visualizations for Telco Customer Churn Dataset
-----------------------------------------------------
Generates common visualizations to understand customer tenure,
churn distribution by various factors, and payment methods.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Configurations
# -----------------------------
DATA_PATH = r"C:/Users/gurio/Downloads/Telco Customer Churn Dataset/WA_Fn-UseC_-Telco-Customer-Churn.xlsx"

# -----------------------------
# Load Data
# -----------------------------
print("Loading dataset...")
df = pd.read_excel(DATA_PATH)

# -----------------------------
# Visualization: Tenure Distribution
# -----------------------------
sns.histplot(df['tenure'], kde=False, bins=30)
plt.title('Customer Tenure Distribution')
plt.xlabel('Tenure (Months)')
plt.ylabel('Count')
plt.show()

# -----------------------------
# Visualization: Churn Counts
# -----------------------------
sns.countplot(x='Churn', data=df)
plt.title('Churn Count')
plt.xlabel('Churn')
plt.ylabel('Number of Customers')
plt.show()

# -----------------------------
# Visualization: Churn Rate by Contract Type
# -----------------------------
contract_churn = df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack()
contract_churn.plot(kind='bar', stacked=True)
plt.title('Churn Rate by Contract Type')
plt.ylabel('Proportion')
plt.xlabel('Contract Type')
plt.legend(title='Churn')
plt.show()

# -----------------------------
# Visualization: Monthly Charges Distribution by Churn Status
# -----------------------------
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='MonthlyCharges', hue='Churn', bins=30, kde=True, element='step')
plt.title('Monthly Charges Distribution by Churn Status')
plt.xlabel('Monthly Charges')
plt.ylabel('Count')
plt.show()

# -----------------------------
# Visualization: Churn Count by Internet Service Type
# -----------------------------
plt.figure(figsize=(7,4))
sns.countplot(x='InternetService', hue='Churn', data=df)
plt.title('Churn Count by Internet Service Type')
plt.xlabel('Internet Service')
plt.ylabel('Number of Customers')
plt.show()

# -----------------------------
# Visualization: Tenure by Churn Status (Boxplot)
# -----------------------------
plt.figure(figsize=(8,5))
sns.boxplot(x='Churn', y='tenure', data=df)
plt.title('Tenure by Churn Status')
plt.xlabel('Churn')
plt.ylabel('Tenure (Months)')
plt.show()

# -----------------------------
# Visualization: Churn Rate by Payment Method
# -----------------------------
plt.figure(figsize=(8,4))
payment_churn = df.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack()
payment_churn.plot(kind='bar', stacked=True)
plt.title('Churn Rate by Payment Method')
plt.ylabel('Proportion')
plt.xlabel('Payment Method')
plt.legend(title='Churn')
plt.show()

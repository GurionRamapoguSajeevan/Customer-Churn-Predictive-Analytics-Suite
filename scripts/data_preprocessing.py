"""
Data Preprocessing Script for Telco Customer Churn Dataset
---------------------------------------------------------
Cleans and encodes data to prepare for machine learning modeling.
"""

import pandas as pd

# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = r"C:/Users/gurio/Downloads/Telco Customer Churn Dataset/WA_Fn-UseC_-Telco-Customer-Churn.xlsx"

# -----------------------------
# Load Data
# -----------------------------
print("Loading dataset...")
df = pd.read_excel(DATA_PATH)

# -----------------------------
# Convert TotalCharges to numeric, handle missing values
# -----------------------------
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print(f"TotalCharges missing before fix: {df['TotalCharges'].isnull().sum()}")

# Replace NaNs in TotalCharges where tenure is 0 with 0
df.loc[df['TotalCharges'].isnull() & (df['tenure'] == 0), 'TotalCharges'] = 0
print(f"TotalCharges missing after fix: {df['TotalCharges'].isnull().sum()}")

print(f"TotalCharges data type: {df['TotalCharges'].dtype}")

# -----------------------------
# Encode Target Variable: Churn
# -----------------------------
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
print("Churn value counts:")
print(df['Churn'].value_counts())


print("Data preprocessing completed successfully.")

# Ensure 'data' directory exists
import os
os.makedirs("data", exist_ok=True)

# Save the processed dataset
df.to_csv("data/processed_clean.csv", index=False)

print("Processed data saved to data/processed_clean.csv")

"""
Basic EDA Script for Telco Customer Churn Dataset
------------------------------------------------
Loads the dataset, inspects structure, missing values,
and prints summary statistics for numerical and categorical variables.
"""

import pandas as pd

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
# Dataset Info
# -----------------------------
print("\n--- Dataset Info ---")
print(df.info())

# -----------------------------
# Target Variable Distribution
# -----------------------------
print("\n--- Churn Value Counts ---")
print(df['Churn'].value_counts())

print("\n--- Churn Distribution (Proportion) ---")
print(df['Churn'].value_counts(normalize=True))

# -----------------------------
# Summary Statistics
# -----------------------------
print("\n--- Summary Stats (Numerical) ---")
print(df.describe())

print("\n--- Summary Stats (Categorical) ---")
print(df.describe(include='object'))  # for categorical variables

# -----------------------------
# Missing Values
# -----------------------------
print("\n--- Missing Values Per Column ---")
print(df.isnull().sum())

print("\nData Types of Columns:")
print(df.dtypes)

print("\nEDA script finished successfully.")

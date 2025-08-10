"""
Feature Engineering for Telco Customer Churn Dataset
----------------------------------------------------
Creates tenure buckets, average monthly charge, total services count.
"""

import pandas as pd

# Load preprocessed data
df = pd.read_csv("data/processed_clean.csv")

# -----------------------------
# Tenure Buckets
# -----------------------------
bins = [0, 12, 24, 48, 60, df['tenure'].max()]
labels = ['<12m', '12-24m', '24-48m', '48-60m', '>60m']
df['tenure_group'] = pd.cut(df['tenure'], bins=bins, labels=labels, include_lowest=True)
df = pd.get_dummies(df, columns=['tenure_group'], drop_first=True)

# -----------------------------
# Average Monthly Charges
# -----------------------------
df['AvgMonthlyCharge'] = df.apply(
    lambda row: row['TotalCharges'] / row['tenure'] if row['tenure'] > 0 else 0,
    axis=1
)

# -----------------------------
# Total Active Services
# -----------------------------
active_service_cols = [
    'PhoneService',
    'MultipleLines_Yes',
    'OnlineSecurity_Yes',
    'OnlineBackup_Yes',
    'DeviceProtection_Yes',
    'TechSupport_Yes',
    'StreamingTV_Yes',
    'StreamingMovies_Yes'
]

for col in active_service_cols:
    if col not in df.columns:
        raise KeyError(f"Expected column '{col}' not found in DataFrame.")

df['TotalServices'] = df[active_service_cols].sum(axis=1)

print("\nTotalServices summary:")
print(df['TotalServices'].describe())
print(df[['TotalServices', 'Churn']].groupby('TotalServices').mean())

# -----------------------------
# Drop Unneeded Columns
# -----------------------------
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

# Save engineered dataset
df.to_csv("data/processed_engineered.csv", index=False)
print("Feature engineering complete. Saved to data/processed_engineered.csv")
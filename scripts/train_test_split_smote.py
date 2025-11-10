"""
Train-Test Split and SMOTE Balancing
------------------------------------
Splits the engineered dataset into train and test sets with stratification,
checks data types, and applies SMOTE oversampling to balance classes in the training set.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# -----------------------------
# Load engineered dataset
# -----------------------------
df = pd.read_csv("data/processed_engineered.csv")

# -----------------------------
# Ensure TotalServices is numeric
# -----------------------------
df['TotalServices'] = pd.to_numeric(df['TotalServices'], errors='coerce')
print(f"TotalServices dtype: {df['TotalServices'].dtype}")
print(f"Missing values in TotalServices: {df['TotalServices'].isnull().sum()}")

# -----------------------------
# Step 8: Split features (X) and target (y)
# -----------------------------
X = df.drop('Churn', axis=1)
y = df['Churn']

# -----------------------------
# Step 9: Train-test split with stratification
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,        # 20% for test set
    random_state=42,      # reproducibility
    stratify=y            # preserve target class proportions
)

# Check resulting class balance
print("\nTraining set churn distribution:")
print(y_train.value_counts(normalize=True))
print("\nTest set churn distribution:")
print(y_test.value_counts(normalize=True))

# -----------------------------
# Apply SMOTE to the training data
# -----------------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Confirm new class balance
print("\nResampled training set class distribution:")
print(y_train_res.value_counts())

# Check dtypes in resampled training set (should all be numeric)
print(X_train_res.dtypes)

# -----------------------------
# (Optional) Save results for model training
# -----------------------------
pd.DataFrame(X_train_res, columns=X.columns).to_csv("data/X_train_smote.csv", index=False)
y_train_res.to_csv("data/y_train_smote.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)


print("\nTrain-test split with SMOTE complete. Files saved in 'data/' folder.")

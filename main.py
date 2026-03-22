import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("loan_data.csv")

# -------------------------------
# HANDLE MISSING VALUES
# -------------------------------
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median(), inplace=True)

# -------------------------------
# DROP COLUMN
# -------------------------------
df.drop('Loan_ID', axis=1, inplace=True)

# -------------------------------
# ENCODING
# -------------------------------
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# -------------------------------
# SPLIT DATA
# -------------------------------
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# RANDOM FOREST MODEL
# -------------------------------
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

# -------------------------------
# SAVE MODEL
# -------------------------------
import pickle

pickle.dump(model, open("loan_model.pkl", "wb"))

print("Random Forest model saved successfully!")
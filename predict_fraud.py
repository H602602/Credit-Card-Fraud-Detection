import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

# Paths
DATA_PATH = "creditcard.csv"
SCALER_PATH = os.path.join("models", "fraud_scaler.pkl")
MODEL_PATH = os.path.join("models", "fraud_detector.pkl")

# Load scaler and model
print("Loading scaler and model...")
scaler = joblib.load(SCALER_PATH)
model = joblib.load(MODEL_PATH)

# Load data
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Separate features and labels
X = df.drop(columns=["Class"])
y = df["Class"]

FEATURE_NAMES = X.columns.tolist()

def predict_row(index: int):
    """
    Predict fraud or not for a given row index from the dataset.
    """
    if index < 0 or index >= len(df):
        raise ValueError(f"Index must be between 0 and {len(df)-1}")

    row_features = X.iloc[index].values.reshape(1, -1)
    row_scaled = scaler.transform(row_features)

    proba = model.predict_proba(row_scaled)[0][1]  # probability of class 1 (fraud)
    pred_label = int(proba >= 0.5)
    actual_label = int(y.iloc[index])

    return actual_label, pred_label, proba

if __name__ == "__main__":
    print("\nCredit Card Fraud Detection - Test a Transaction")
    print(f"Total rows in dataset: {len(df)}")

    while True:
        user_input = input(f"\nEnter a row index between 0 and {len(df)-1} (or 'q' to quit): ")

        if user_input.lower() == 'q':
            break

        try:
            idx = int(user_input)
            actual, pred, p = predict_row(idx)

            actual_str = "FRAUD" if actual == 1 else "NORMAL"
            pred_str = "FRAUD" if pred == 1 else "NORMAL"

            print(f"\nTransaction index: {idx}")
            print(f"Actual label: {actual_str}")
            print(f"Predicted : {pred_str}")
            print(f"Fraud probability: {p:.4f}")

        except ValueError as e:
            print("Error:", e)
        except Exception as e:
            print("Unexpected error:", e)

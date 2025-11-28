import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE  # from imbalanced-learn


# -----------------------------
# 1. LOAD DATA
# -----------------------------
DATA_PATH = "creditcard.csv"

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print(df.head())


# -----------------------------
# 2. SEPARATE FEATURES AND LABEL
# -----------------------------
# 'Class' is the label: 0 = normal, 1 = fraud
X = df.drop(columns=["Class"])
y = df["Class"]

print("\nClass distribution in full data:")
print(y.value_counts())   # majority will be 0 (non-fraud)


# -----------------------------
# 3. TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y   # keeps same class ratio in train & test
)

print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)
print("Train class distribution:")
print(y_train.value_counts())
print("Test class distribution:")
print(y_test.value_counts())


# -----------------------------
# 4. FEATURE SCALING
# -----------------------------
# We scale features so that they have similar ranges
# Important for Logistic Regression
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -----------------------------
# 5. HANDLE IMBALANCE WITH SMOTE
# -----------------------------
# SMOTE = Synthetic Minority Oversampling Technique
# It will create synthetic fraud samples to balance the classes in training data.

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print("\nAfter SMOTE, train class distribution:")
print(np.bincount(y_train_resampled))
# index 0 = count of class 0, index 1 = count of class 1


# -----------------------------
# 6. TRAIN MODEL
# -----------------------------
model = LogisticRegression(
    max_iter=1000,
    n_jobs=-1
)

print("\nTraining Logistic Regression model...")
model.fit(X_train_resampled, y_train_resampled)


# -----------------------------
# 7. EVALUATE MODEL
# -----------------------------
y_pred = model.predict(X_test_scaled)
y_scores = model.predict_proba(X_test_scaled)[:, 1]  # probability of class 1 (fraud)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

roc_auc = roc_auc_score(y_test, y_scores)
print("ROC-AUC Score:", roc_auc)


# -----------------------------
# 8. SAVE MODEL + SCALER
# -----------------------------
os.makedirs("models", exist_ok=True)

scaler_path = os.path.join("models", "fraud_scaler.pkl")
model_path = os.path.join("models", "fraud_detector.pkl")

joblib.dump(scaler, scaler_path)
joblib.dump(model, model_path)

print(f"\nScaler saved to: {scaler_path}")
print(f"Model saved to:  {model_path}")
print("\nTraining complete.")

#!/usr/bin/env python3
"""
Standard Logistic Regression on Breast Cancer dataset
(no animation, just training and evaluation).
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# 1. Load dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

print("Features shape:", X.shape)
print("Classes:", np.unique(y))

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Preprocess (standardize features)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 4. Logistic regression model
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_scaled, y_train)

# 5. Predictions
y_pred = clf.predict(X_test_scaled)
y_proba = clf.predict_proba(X_test_scaled)[:,1]

# 6. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 7. Coefficients (interpreting importance)
coef = pd.Series(clf.coef_[0], index=feature_names)
print("\nTop 10 features by magnitude:")
print(coef.abs().sort_values(ascending=False).head(10))

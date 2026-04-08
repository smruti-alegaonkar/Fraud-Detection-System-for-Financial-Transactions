"""
preprocessing.py  — PaySim version
------------------------------------
Stateful preprocessing pipeline.

fit_transform() trains StandardScaler on training data only.
transform()     applies fitted scaler to new data (no leakage).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

FEATURE_COLS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "type_CASH_OUT",
    "type_TRANSFER",
    "type_PAYMENT",
    "type_CASH_IN",
    "type_DEBIT",
    "balance_diff_orig",
    "balance_diff_dest",
    "orig_balance_zero",
    "dest_balance_zero",
    "amount_ratio_orig",
    "step",
]

TARGET_COL = "Class"

# Only continuous columns need scaling; binary flags do not
_SCALE_COLS = [
    "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "balance_diff_orig", "balance_diff_dest",
    "amount_ratio_orig", "step",
]


class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_cols = FEATURE_COLS
        self.is_fitted = False

    def load_data(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        print(f"Loaded {len(df):,} rows x {len(df.columns)} columns")
        missing = df.isnull().sum().sum()
        if missing > 0:
            for col in df.select_dtypes(include=np.number).columns:
                df[col].fillna(df[col].median(), inplace=True)
        return df

    def split(self, df: pd.DataFrame, test_size: float = 0.20,
              random_state: int = 42):
        X = df[self.feature_cols].copy()
        y = df[TARGET_COL].copy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
        print(f"Train fraud rate: {y_train.mean():.4%}")
        print(f"Test  fraud rate: {y_test.mean():.4%}")
        return X_train, X_test, y_train, y_test

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        X = X_train.copy()
        X[_SCALE_COLS] = self.scaler.fit_transform(X[_SCALE_COLS])
        self.is_fitted = True
        return X.values

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Call fit_transform() first.")
        X = X.copy()
        X[_SCALE_COLS] = self.scaler.transform(X[_SCALE_COLS])
        return X.values

    def save(self, path: str = "models/preprocessor.joblib"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"Preprocessor saved to {path}")

    @staticmethod
    def load(path: str = "models/preprocessor.joblib") -> "Preprocessor":
        return joblib.load(path)

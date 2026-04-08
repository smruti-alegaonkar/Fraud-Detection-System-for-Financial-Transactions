"""
data_generator.py  — PaySim dataset version
--------------------------------------------
Loads the PaySim financial transactions dataset (fraud.csv).

Dataset facts:
  - 6,362,620 transactions (mobile money simulation)
  - 8,213 fraud cases  (~0.13%)
  - Columns: step, type, amount, nameOrig, oldbalanceOrg,
             newbalanceOrig, nameDest, oldbalanceDest,
             newbalanceDest, isFraud, isFlaggedFraud

Key insight: fraud ONLY occurs in TRANSFER and CASH_OUT transactions.
We engineer several powerful features from the raw columns.
"""

import numpy as np
import pandas as pd
import os

DATASET_FILE = "fraud.csv"

# These are the engineered features our model will use
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
    "balance_diff_orig",    # how much sender's balance changed
    "balance_diff_dest",    # how much receiver's balance changed
    "orig_balance_zero",    # sender balance wiped to zero (fraud signal)
    "dest_balance_zero",    # receiver had zero balance before (mule account)
    "amount_ratio_orig",    # amount / sender original balance
    "step",                 # time step (hour proxy)
]


def load_kaggle_dataset(data_dir: str = "data") -> pd.DataFrame:
    """
    Load fraud.csv and engineer features for the model.
    """
    path = os.path.join(data_dir, DATASET_FILE)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n\nDataset not found at: {path}\n\n"
            "  Place your fraud.csv inside the data/ folder.\n"
        )

    print(f"Loading {path} ...")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df):,} rows x {len(df.columns)} columns")

    # ── Missing values ───────────────────────────────────────────────────────
    missing = df.isnull().sum().sum()
    print(f"  Missing values: {missing}")

    # ── Rename target column ─────────────────────────────────────────────────
    # Our pipeline expects 'Class' (0=legit, 1=fraud)
    df.rename(columns={"isFraud": "Class"}, inplace=True)

    # ── One-hot encode transaction type ──────────────────────────────────────
    for t in ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"]:
        df[f"type_{t}"] = (df["type"] == t).astype(int)

    # ── Feature engineering ──────────────────────────────────────────────────
    # How much did sender's balance actually change?
    df["balance_diff_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]

    # How much did receiver's balance change?
    df["balance_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]

    # Key fraud signal: sender balance wiped to exactly zero
    df["orig_balance_zero"] = (df["newbalanceOrig"] == 0).astype(int)

    # Key fraud signal: destination was a fresh/empty account (mule)
    df["dest_balance_zero"] = (df["oldbalanceDest"] == 0).astype(int)

    # Ratio of transaction amount to sender's original balance
    df["amount_ratio_orig"] = df["amount"] / (df["oldbalanceOrg"] + 1)

    # ── Drop columns not used by model ──────────────────────────────────────
    df.drop(columns=["type", "nameOrig", "nameDest", "isFlaggedFraud"],
            inplace=True)

    # ── Print class distribution ─────────────────────────────────────────────
    n_fraud = df["Class"].sum()
    n_legit = len(df) - n_fraud
    print(f"\n  Legitimate : {n_legit:,}  ({n_legit/len(df):.4%})")
    print(f"  Fraudulent : {n_fraud:,}  ({n_fraud/len(df):.4%})")
    print(f"  Imbalance  : 1 fraud per {n_legit//n_fraud:,} legit transactions\n")

    return df


def save_dataset(df: pd.DataFrame, output_dir: str = "data") -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "transactions_prepared.csv")
    df.to_csv(path, index=False)
    print(f"Prepared dataset saved to {path}")
    return path


# Alias so train.py import works unchanged
def generate_fraud_dataset(data_dir: str = "data", **kwargs) -> pd.DataFrame:
    return load_kaggle_dataset(data_dir=data_dir)


if __name__ == "__main__":
    df = load_kaggle_dataset(data_dir="../data")
    print(df[FEATURE_COLS + ["Class"]].head())

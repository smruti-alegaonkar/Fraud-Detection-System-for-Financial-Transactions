import os, sys, json
import pandas as pd
import joblib

BASE_DIR  = r"c:\Users\shari\PersonalDevelopment\Fraud-Detection-System-for-Financial-Transactions"
sys.path.insert(0, BASE_DIR)
from src.data_generator import load_kaggle_dataset

MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR  = os.path.join(BASE_DIR, "data")

with open(os.path.join(MODEL_DIR, "config.json")) as f:
    CONFIG = json.load(f)

print("Loading Models & Data...")
PREPROCESSOR = joblib.load(os.path.join(MODEL_DIR, CONFIG["preprocessor"]))
RF_MODEL     = joblib.load(os.path.join(MODEL_DIR, CONFIG["best_model"]))
LR_MODEL     = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.joblib"))
FEATURE_COLS = CONFIG["feature_cols"]

df = load_kaggle_dataset(data_dir=DATA_DIR)
print("Sampling 500k transactions for inference...\n")
sample_df = df.sample(n=500000, random_state=42).copy()
X = PREPROCESSOR.transform(sample_df[FEATURE_COLS])

def print_cases(df_pool, category_name, num=3):
    print(f"   {category_name.upper()}")
    print(f"   ----------------------------------")
    for i, (_, row) in enumerate(df_pool.head(num).iterrows()):
        txn_type = "TRANSFER" if row["type_TRANSFER"] == 1 else "CASH_OUT" if row["type_CASH_OUT"] == 1 else "PAYMENT"
        print(f"   * TestCase {i+1}: Output = {row['proba']*100:.2f}% | {txn_type.ljust(9)} | Amt: ${row['amount']:<7.0f} | Sender: ${row['oldbalanceOrg']:<7.0f} -> ${row['newbalanceOrig']:<7.0f} | Dest: ${row['oldbalanceDest']:<7.0f} -> ${row['newbalanceDest']:<7.0f}")
    print()


# ==========================================
# 1. RANDOM FOREST TEST CASES
# ==========================================
print("=========================================================================")
print("                      1. RANDOM FOREST TEST CASES                        ")
print("=========================================================================\n")

sample_df['proba'] = RF_MODEL.predict_proba(X)[:, 1]

# Random forest boundaries:
rf_legit  = sample_df[sample_df['proba'] < 0.015]
rf_medium = sample_df[(sample_df['proba'] >= 0.05) & (sample_df['proba'] <= 0.85)]
rf_fraud  = sample_df[sample_df['proba'] >= 0.90]

if len(rf_legit) > 0: print_cases(rf_legit, "✅ Looks Legitimate (LOW Risk)", 3)
if len(rf_medium) > 0: print_cases(rf_medium, "⚠️ Needs Review (MEDIUM Risk)", 3)
if len(rf_fraud) > 0: print_cases(rf_fraud, "🚨 Fraud Detected (HIGH Risk)", 3)


# ==========================================
# 2. LOGISTIC REGRESSION (CUSTOM) TEST CASES
# ==========================================
print("\n=========================================================================")
print("                2. LOGISTIC REGRESSION (CUSTOM) TEST CASES               ")
print("=========================================================================\n")

sample_df['proba'] = LR_MODEL.predict_proba(X)[:, 1]

# Dynamic boundaries because Custom LR clusters differently based on exact iterations / convergence
median_proba = sample_df['proba'].median()
max_proba = sample_df['proba'].max()
min_proba = sample_df['proba'].min()

print(f"[Model Diagnostic] Min Prob: {min_proba:.4f}, Median: {median_proba:.4f}, Max: {max_proba:.4f}\n")

lr_legit  = sample_df[sample_df['proba'] < median_proba * 1.05].sort_values(by="proba")
lr_medium = sample_df[(sample_df['proba'] >= median_proba * 1.05) & (sample_df['proba'] <= max_proba * 0.90)]
lr_fraud  = sample_df.sort_values(by="proba", ascending=False)

if len(lr_legit) > 0: print_cases(lr_legit, "✅ Looks Legitimate (LOW Risk)", 3)
if len(lr_medium) > 0: print_cases(lr_medium, "⚠️ Needs Review (MEDIUM Risk)", 3)
if len(lr_fraud) > 0: print_cases(lr_fraud, "🚨 Fraud Detected (HIGH Risk)", 3)
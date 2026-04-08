"""
train.py — PaySim dataset pipeline
------------------------------------
Run this once to train all models and generate all charts.

    cd fraud_detection
    python src/train.py
"""

import os, sys, json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generator  import generate_fraud_dataset, FEATURE_COLS
from src.preprocessing   import Preprocessor
from src.imbalance_handler import smote_oversample, random_undersample
from src.models import (
    train_logistic_regression, train_random_forest, train_isolation_forest,
    evaluate_classifier, evaluate_isolation_forest,
    find_optimal_threshold, save_model
)
from src.visualizations import (
    plot_class_distribution, plot_feature_distributions,
    plot_correlation_heatmap, plot_roc_curves, plot_pr_curves,
    plot_confusion_matrix, plot_threshold_analysis,
    plot_feature_importance, plot_business_scenarios,
    plot_model_comparison
)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "static/plots")
os.makedirs(REPORT_DIR, exist_ok=True)

def banner(msg):
    print(f"\n{'='*65}\n  {msg}\n{'='*65}")


# ── STEP 1: Load dataset ─────────────────────────────────────────────────────
banner("STEP 1 — Loading PaySim Dataset (fraud.csv)")
df = generate_fraud_dataset(data_dir=DATA_DIR)
print(f"Dataset shape: {df.shape}")
print(f"Features: {FEATURE_COLS}")

# ── STEP 2: EDA charts ───────────────────────────────────────────────────────
banner("STEP 2 — EDA & Visualizations")
plot_class_distribution(df["Class"], title="Class Distribution — PaySim Dataset")

eda_features = [
    "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "balance_diff_orig",
    "orig_balance_zero", "dest_balance_zero",
    "type_TRANSFER", "type_CASH_OUT", "amount_ratio_orig",
    "step", "balance_diff_dest"
]
plot_feature_distributions(df, eda_features)
plot_correlation_heatmap(df, FEATURE_COLS + ["Class"])
print("EDA charts saved to reports/")

# ── STEP 3: Preprocessing ────────────────────────────────────────────────────
banner("STEP 3 — Preprocessing")
prep = Preprocessor()
X_train_df, X_test_df, y_train, y_test = prep.split(df)
X_train = prep.fit_transform(X_train_df)
X_test  = prep.transform(X_test_df)
prep.save(os.path.join(MODEL_DIR, "preprocessor.joblib"))
y_train_arr = y_train.values
y_test_arr  = y_test.values

# ── STEP 4: Logistic Regression baseline ─────────────────────────────────────
banner("STEP 4 — Logistic Regression (baseline, class_weight='balanced')")
lr_model = train_logistic_regression(X_train, y_train_arr)
lr_result = evaluate_classifier(lr_model, X_test, y_test_arr,
                                 "Logistic Regression", threshold=0.5)
save_model(lr_model, os.path.join(MODEL_DIR, "logistic_regression.joblib"))

# ── STEP 5: Random Forest ────────────────────────────────────────────────────
banner("STEP 5 — Random Forest")
rf_model = train_random_forest(X_train, y_train_arr)
rf_result = evaluate_classifier(rf_model, X_test, y_test_arr,
                                 "Random Forest", threshold=0.5)
save_model(rf_model, os.path.join(MODEL_DIR, "random_forest.joblib"))

# ── STEP 6: Isolation Forest ─────────────────────────────────────────────────
banner("STEP 6 — Isolation Forest (Anomaly Detection)")
fraud_rate = y_train_arr.mean()
iso_model = train_isolation_forest(X_train, contamination=fraud_rate)
iso_result = evaluate_isolation_forest(iso_model, X_test, y_test_arr)
save_model(iso_model, os.path.join(MODEL_DIR, "isolation_forest.joblib"))

# ── STEP 7: SMOTE ────────────────────────────────────────────────────────────
banner("STEP 7 — SMOTE Resampling -> Retrain Logistic Regression")
X_smote, y_smote = smote_oversample(X_train, y_train_arr,
                                     sampling_ratio=0.10, random_state=42)
lr_smote = train_logistic_regression(X_smote, y_smote, class_weight=None)
lr_smote_result = evaluate_classifier(lr_smote, X_test, y_test_arr,
                                       "LR + SMOTE", threshold=0.5)
save_model(lr_smote, os.path.join(MODEL_DIR, "lr_smote.joblib"))

# ── STEP 8: Threshold tuning ─────────────────────────────────────────────────
banner("STEP 8 — Threshold Tuning on Random Forest")
optimal_t = find_optimal_threshold(y_test_arr, rf_result["y_proba"], beta=2.0)
rf_result_tuned = evaluate_classifier(rf_model, X_test, y_test_arr,
                                       "RF (tuned threshold)", threshold=optimal_t)
save_model(rf_model, os.path.join(MODEL_DIR, "rf_best.joblib"))
plot_threshold_analysis(y_test_arr, rf_result["y_proba"],
                        model_name="Random Forest",
                        optimal_threshold=optimal_t)

# ── STEP 9: Charts ───────────────────────────────────────────────────────────
banner("STEP 9 — Evaluation Charts")
all_results = [lr_result, rf_result, iso_result, lr_smote_result]
plot_roc_curves(all_results, y_test_arr)
plot_pr_curves(all_results, y_test_arr)
plot_confusion_matrix(y_test_arr, rf_result_tuned["y_pred"],
                      "Random Forest", optimal_t)
plot_feature_importance(rf_model, FEATURE_COLS, "Random Forest")
plot_business_scenarios(y_test_arr, rf_result["y_proba"])
plot_model_comparison(all_results)



# ── STEP 10: Summary ─────────────────────────────────────────────────────────
banner("STEP 10 — Results Summary")
summary = []
for r in [lr_result, lr_smote_result, rf_result, rf_result_tuned, iso_result]:
    summary.append({
        "Model": r["model"], "Threshold": r["threshold"],
        "Precision": f"{r['precision']:.4f}", "Recall": f"{r['recall']:.4f}",
        "F1": f"{r['f1']:.4f}", "ROC-AUC": f"{r['roc_auc']:.4f}",
        "PR-AUC": f"{r['pr_auc']:.4f}",
        "TP": r["tp"], "FP": r["fp"], "FN": r["fn"], "TN": r["tn"],
    })
summary_df = pd.DataFrame(summary)
print("\n" + summary_df.to_string(index=False))
summary_df.to_csv(os.path.join(REPORT_DIR, "model_comparison.csv"), index=False)

# Save config for Flask app
config = {
    "optimal_threshold": optimal_t,
    "feature_cols": FEATURE_COLS,
    "best_model": "random_forest.joblib",
    "preprocessor": "preprocessor.joblib",
    "dataset": "PaySim (fraud.csv)",
}
with open(os.path.join(MODEL_DIR, "config.json"), "w") as f:
    json.dump(config, f, indent=2)
print(f"\nConfig saved to models/config.json")

banner("Training Complete!")
print(f"  Models  -> {MODEL_DIR}/")
print(f"  Reports -> {REPORT_DIR}/")
print(f"\n  Next: run  python app/app.py  and open http://127.0.0.1:5000")

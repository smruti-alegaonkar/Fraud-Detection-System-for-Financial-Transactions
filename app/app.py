"""
app.py — PaySim dataset version
--------------------------------
Flask web app for fraud detection using the PaySim dataset.

Routes:
  GET  /          -> Prediction UI
  POST /predict   -> Single prediction (JSON)
  GET  /simulate  -> Live simulation page
  POST /simulate  -> Batch simulation
  GET  /dashboard -> Model metrics
  GET  /health    -> Health check
"""

import os, sys, json, time, random
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "app", "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

print("Loading models...")
with open(os.path.join(MODEL_DIR, "config.json")) as f:
    CONFIG = json.load(f)

PREPROCESSOR = joblib.load(os.path.join(MODEL_DIR, CONFIG["preprocessor"]))
RF_MODEL     = joblib.load(os.path.join(MODEL_DIR, CONFIG["best_model"]))
LR_MODEL     = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.joblib"))
OPTIMAL_T    = CONFIG["optimal_threshold"]
FEATURE_COLS = CONFIG["feature_cols"]
print(f"Models loaded. Optimal threshold: {OPTIMAL_T:.4f}")

TRANSACTION_TYPES = ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"]


def build_features(data: dict) -> pd.DataFrame:
    """
    Convert raw user input (with 'type' dropdown) into model feature vector.
    Applies the same feature engineering as data_generator.py.
    """
    txn_type   = data.get("type", "PAYMENT")
    amount     = float(data.get("amount", 0))
    oldbal_o   = float(data.get("oldbalanceOrg", 0))
    newbal_o   = float(data.get("newbalanceOrig", 0))
    oldbal_d   = float(data.get("oldbalanceDest", 0))
    newbal_d   = float(data.get("newbalanceDest", 0))
    step       = float(data.get("step", 1))

    row = {
        "amount":            amount,
        "oldbalanceOrg":     oldbal_o,
        "newbalanceOrig":    newbal_o,
        "oldbalanceDest":    oldbal_d,
        "newbalanceDest":    newbal_d,
        "type_CASH_OUT":     int(txn_type == "CASH_OUT"),
        "type_TRANSFER":     int(txn_type == "TRANSFER"),
        "type_PAYMENT":      int(txn_type == "PAYMENT"),
        "type_CASH_IN":      int(txn_type == "CASH_IN"),
        "type_DEBIT":        int(txn_type == "DEBIT"),
        "balance_diff_orig": oldbal_o - newbal_o,
        "balance_diff_dest": newbal_d - oldbal_d,
        "orig_balance_zero": int(newbal_o == 0),
        "dest_balance_zero": int(oldbal_d == 0),
        "amount_ratio_orig": amount / (oldbal_o + 1),
        "step":              step,
    }
    return pd.DataFrame([row])[FEATURE_COLS]


def predict_transaction(data: dict, model_name: str = "rf",
                         threshold: float = None) -> dict:
    t_start = time.time()
    row_df = build_features(data)
    X = PREPROCESSOR.transform(row_df)

    model = RF_MODEL if model_name == "rf" else LR_MODEL
    t = threshold if threshold is not None else OPTIMAL_T

    proba = model.predict_proba(X)[0, 1]
    pred  = int(proba >= t)

    if proba < 0.1:   risk = "LOW"
    elif proba < 0.4: risk = "MEDIUM"
    elif proba < t:   risk = "HIGH"
    else:             risk = "FRAUD"

    # Fraud flags using PaySim domain knowledge
    flags = []
    txn_type  = data.get("type", "PAYMENT")
    amount    = float(data.get("amount", 0))
    oldbal_o  = float(data.get("oldbalanceOrg", 0))
    newbal_o  = float(data.get("newbalanceOrig", 0))
    oldbal_d  = float(data.get("oldbalanceDest", 0))

    if txn_type in ("TRANSFER", "CASH_OUT"):
        flags.append(f"Transaction type '{txn_type}' — only type where fraud occurs")
    if newbal_o == 0 and amount > 0:
        flags.append("Sender balance wiped to zero after transaction")
    if oldbal_d == 0 and txn_type in ("TRANSFER", "CASH_OUT"):
        flags.append("Destination was an empty account (possible mule account)")
    if oldbal_o > 0 and amount / oldbal_o > 0.9:
        flags.append(f"Amount is {amount/oldbal_o:.0%} of sender's balance (nearly full drain)")
    if amount > 1_000_000:
        flags.append(f"Very large transaction: ${amount:,.2f}")
    if abs((oldbal_o - newbal_o) - amount) > 1:
        flags.append("Balance change does not match transaction amount (inconsistency)")

    latency_ms = round((time.time() - t_start) * 1000, 2)
    return {
        "prediction":    pred,
        "label":         "FRAUD" if pred == 1 else "LEGITIMATE",
        "probability":   round(float(proba), 6),
        "probability_pct": round(float(proba) * 100, 3),
        "risk_level":    risk,
        "threshold_used": round(t, 4),
        "model_used":    model_name,
        "flags":         flags,
        "latency_ms":    latency_ms,
    }


@app.route("/")
def index():
    return render_template("index.html",
                           optimal_threshold=round(OPTIMAL_T, 4),
                           txn_types=TRANSACTION_TYPES)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    try:
        threshold  = float(data.get("threshold", OPTIMAL_T))
        model_name = data.get("model", "rf")
        result = predict_transaction(data, model_name, threshold)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/simulate", methods=["GET", "POST"])
def simulate():
    if request.method == "GET":
        return render_template("simulate.html",
                               optimal_threshold=round(OPTIMAL_T, 4))

    n = int(request.json.get("n", 1))
    inject_fraud = request.json.get("inject_fraud", True)
    results = []
    rng = np.random.default_rng()

    for _ in range(n):
        is_suspicious = inject_fraud and (rng.random() < 0.06)

        if is_suspicious:
            txn_type = rng.choice(["TRANSFER", "CASH_OUT"])
            amount   = float(rng.uniform(50_000, 500_000))
            old_o    = amount  # sender had exactly that amount — wipes to zero
            new_o    = 0.0
            old_d    = 0.0     # fresh mule account
            new_d    = amount
        else:
            txn_type = rng.choice(TRANSACTION_TYPES,
                                   p=[0.35, 0.34, 0.22, 0.08, 0.01])
            amount   = float(rng.lognormal(10, 1.5))
            old_o    = float(rng.uniform(amount, amount * 5))
            new_o    = old_o - amount
            old_d    = float(rng.uniform(0, 100_000))
            new_d    = old_d + amount

        data = {
            "type": txn_type, "amount": round(amount, 2),
            "oldbalanceOrg": round(old_o, 2), "newbalanceOrig": round(new_o, 2),
            "oldbalanceDest": round(old_d, 2), "newbalanceDest": round(new_d, 2),
            "step": int(rng.integers(1, 743)),
        }
        result = predict_transaction(data)
        results.append({
            "id":     f"TXN-{random.randint(100000,999999)}",
            "type":   txn_type,
            "amount": round(amount, 2),
            "hour":   data["step"],
            **result,
        })

    return jsonify(results)


@app.route("/dashboard")
def dashboard():
    csv_path = os.path.join(BASE_DIR, "reports", "model_comparison.csv")
    try:
        df = pd.read_csv(csv_path)
        metrics = df.to_dict(orient="records")
    except FileNotFoundError:
        metrics = []
    return render_template("dashboard.html",
                           metrics=metrics,
                           optimal_threshold=round(OPTIMAL_T, 4))


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "dataset": CONFIG.get("dataset", "PaySim"),
        "optimal_threshold": OPTIMAL_T,
        "feature_count": len(FEATURE_COLS),
    })


if __name__ == "__main__":
    print("\nFraudShield starting — http://127.0.0.1:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)

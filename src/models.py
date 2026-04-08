"""
models.py
---------
Model definitions and training wrappers.

Models implemented:
  1. LogisticRegression  (baseline, class_weight='balanced')
  2. RandomForestClassifier (stronger, handles imbalance well)
  3. IsolationForest (anomaly detection — unsupervised)

Key concept — WHY class_weight='balanced'?
   sklearn computes weight = n_samples / (n_classes * n_class_samples)
   Frauds get ~590× more weight in the loss, so misclassifying a fraud
   hurts ~590× more than misclassifying a legit transaction.
"""

import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, roc_curve
)


# ── Utility ─────────────────────────────────────────────────────────────────

def print_banner(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


# ── Logistic Regression ──────────────────────────────────────────────────────

def train_logistic_regression(X_train, y_train,
                               class_weight="balanced",
                               random_state=42) -> LogisticRegression:
    """
    Baseline model.

    class_weight='balanced' → adjusts weights inversely proportional to
    class frequencies.  Equivalent to over-sampling fraud ~590× in the loss.
    """
    print_banner("Training Logistic Regression")
    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs",
        C=0.1,          # Regularisation — prevents overfitting
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print(f"   ✅ Converged | Coefficients shape: {model.coef_.shape}")
    return model


# ── Random Forest ────────────────────────────────────────────────────────────

def train_random_forest(X_train, y_train,
                        class_weight="balanced",
                        random_state=42) -> RandomForestClassifier:
    """
    Ensemble tree model. Generally outperforms LR on tabular fraud data.

    class_weight='balanced_subsample' → applies balancing per bootstrap sample,
    which is slightly better than 'balanced' for forests.
    """
    print_banner("Training Random Forest")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print(f"   ✅ Trained {model.n_estimators} trees")
    return model


# ── Isolation Forest (Anomaly Detection) ─────────────────────────────────────

def train_isolation_forest(X_train,
                            contamination: float = 0.0017,
                            random_state: int = 42) -> IsolationForest:
    """
    Unsupervised anomaly detection.

    Isolation Forest isolates anomalies (fraud) by randomly selecting a feature
    and a split value.  Anomalies require fewer splits → shorter average path.

    contamination = expected fraction of anomalies (our fraud rate ~0.17%)

    Note: IsolationForest does NOT use labels during training — it's
    unsupervised.  We compare it to supervised models to show the difference.
    """
    print_banner("Training Isolation Forest (Anomaly Detection)")
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train)   # ← no y_train — purely unsupervised
    print("   ✅ Isolation Forest fitted (unsupervised)")
    return model


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_classifier(model, X_test, y_test,
                         model_name: str = "Model",
                         threshold: float = 0.5) -> dict:
    """
    Full evaluation suite for a supervised classifier.

    Returns a dict of all metrics for comparison tables.
    """
    print_banner(f"Evaluating: {model_name} (threshold={threshold})")

    # Raw probabilities (score for class=1)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Apply threshold (default 0.5 → anything above → fraud)
    y_pred = (y_proba >= threshold).astype(int)

    # ── Print report ────────────────────────────────────────────────────────
    print(classification_report(y_test, y_pred,
                                 target_names=["Legit", "Fraud"],
                                 digits=4))

    roc_auc = roc_auc_score(y_test, y_proba)
    avg_prec = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)

    print(f"   ROC-AUC : {roc_auc:.4f}")
    print(f"   Avg Precision (PR-AUC): {avg_prec:.4f}")
    print(f"   Confusion Matrix:\n   TN={tn:,}  FP={fp:,}\n   FN={fn:,}  TP={tp:,}")

    # Business interpretation
    print(f"\n   💼 Business Impact:")
    print(f"      Fraud caught:    {tp:,} / {tp+fn:,} ({recall:.1%} recall)")
    print(f"      False alarms:    {fp:,}  (analysts must review these)")
    print(f"      Missed fraud:    {fn:,}  (financial loss risk)")

    return {
        "model": model_name,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": avg_prec,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "y_proba": y_proba,
        "y_pred": y_pred,
    }


def evaluate_isolation_forest(model: IsolationForest,
                               X_test: np.ndarray,
                               y_test: np.ndarray) -> dict:
    """
    Evaluate Isolation Forest.

    IsolationForest.predict() returns +1 (normal) or -1 (anomaly).
    We convert: -1 → fraud=1, +1 → legit=0
    decision_function() returns anomaly scores (lower = more anomalous).
    """
    print_banner("Evaluating: Isolation Forest")

    raw_pred = model.predict(X_test)              # +1 or -1
    y_pred = np.where(raw_pred == -1, 1, 0)       # convert to 0/1

    # Anomaly score (negate so higher = more anomalous = more fraud-like)
    scores = -model.decision_function(X_test)

    roc_auc = roc_auc_score(y_test, scores)
    avg_prec = average_precision_score(y_test, scores)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)

    print(classification_report(y_test, y_pred,
                                 target_names=["Legit", "Fraud"], digits=4))
    print(f"   ROC-AUC : {roc_auc:.4f}")
    print(f"   PR-AUC  : {avg_prec:.4f}")

    return {
        "model": "Isolation Forest",
        "threshold": "auto",
        "precision": precision, "recall": recall, "f1": f1,
        "roc_auc": roc_auc, "pr_auc": avg_prec,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "y_proba": scores,
        "y_pred": y_pred,
    }


# ── Threshold Tuning ─────────────────────────────────────────────────────────

def find_optimal_threshold(y_test: np.ndarray,
                            y_proba: np.ndarray,
                            beta: float = 2.0) -> float:
    """
    Find the threshold that maximises F-beta score.

    beta > 1 → weights recall more than precision (good for fraud: catch more)
    beta < 1 → weights precision more (fewer false alarms)
    beta = 1 → standard F1

    For fraud detection, beta=2 means catching fraud is 2× more important
    than avoiding false alarms.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

    # F-beta = (1 + beta²) × (P × R) / (beta² × P + R)
    b2 = beta ** 2
    f_betas = ((1 + b2) * precisions[:-1] * recalls[:-1]
               / (b2 * precisions[:-1] + recalls[:-1] + 1e-9))

    best_idx = np.argmax(f_betas)
    best_threshold = thresholds[best_idx]
    best_fbeta = f_betas[best_idx]

    print(f"\n🎯 Optimal threshold (F{beta}): {best_threshold:.4f}  "
          f"(F{beta}={best_fbeta:.4f}, "
          f"P={precisions[best_idx]:.4f}, R={recalls[best_idx]:.4f})")
    return float(best_threshold)


# ── Save / Load ──────────────────────────────────────────────────────────────

def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"💾 Model saved → {path}")


def load_model(path: str):
    return joblib.load(path)

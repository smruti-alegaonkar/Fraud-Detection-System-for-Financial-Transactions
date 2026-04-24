"""
models.py
---------
Model definitions and training wrappers.

Models implemented:
  1. LogisticRegressionFromScratch  (baseline, class_weight='balanced')
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
# from sklearn.linear_model import LogisticRegression
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


# ── Logistic Regression
class LogisticRegressionFromScratch:
    def __init__(self, learning_rate=0.05, num_iterations=2000):
        # The step size for Gradient Descent
        self.learning_rate = learning_rate
        # How many times we pass through the data to optimize weights
        self.num_iterations = num_iterations
        
        # W represents feature weights (how important is 'amount' vs 'step'?)
        self.weights = None
        # B represents the bias (base probability of fraud)
        self.bias = None
        self.classes_ = np.array([0, 1])

    def _sigmoid(self, z):
        """
        The core activation function.
        It maps any real number into a probability between 0 and 1.
        Formula: 1 / (1 + e^-z)
        """
        # np.clip prevents mathematical overflow on very massive numbers 
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Gradient Descent Optimization with Native Class Balancing
        """
        num_samples, num_features = X.shape
        
        # Calculate class weights manually to handle 99.8% Legitimate imbalance
        num_fraud = np.sum(y == 1)
        num_legit = num_samples - num_fraud

        # Formula: weight = n_samples / (n_classes * n_samples_in_class)
        # This penalizes the model 590x more for missing a Fraud!
        weight_legit = num_samples / (2.0 * num_legit)
        weight_fraud = num_samples / (2.0 * num_fraud)

        # Create a vectorized array where each transaction has its appropriate weight
        sample_weights = np.where(y == 1, weight_fraud, weight_legit)
        
        # 1. Initialize weights and bias to zeros
        self.weights = np.zeros(num_features)
        self.bias = 0

        # 2. Gradient Descent Loop
        for i in range(self.num_iterations):
            
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # Multiply the raw error by the sample weight
            error = (y_predicted - y) * sample_weights

            dw = (1 / num_samples) * np.dot(X.T, error)
            db = (1 / num_samples) * np.sum(error)

            # Step C: Update Weights (Move in the direction of 'less wrong')
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # (Optional) Print loss every 100 iterations to watch it learn
            if i % 100 == 0:
                loss = self._binary_cross_entropy(y, y_predicted)
                print(f"Iteration {i}: Loss = {loss:.4f}")

    def predict_proba(self, X):
        """Returns the exact percentage probability of Fraud"""
        linear_model = np.dot(X, self.weights) + self.bias
        probas = self._sigmoid(linear_model)
        # Scikit-learn expects a 2D array of both class probabilities
        return np.vstack([1 - probas, probas]).T
        
    def predict(self, X, threshold=0.5):
        """Returns a hard 1 (Fraud) or 0 (Legitimate)"""
        linear_model = np.dot(X, self.weights) + self.bias
        probabilities = self._sigmoid(linear_model)
        return (probabilities >= threshold).astype(int)

    def _binary_cross_entropy(self, y_true, y_pred):
        """The Log-Loss function"""
        epsilon = 1e-9
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

def train_logistic_regression(X_train, y_train, class_weight="balanced", random_state=42):
    """
    Wrapper function that trains our custom LogisticRegressionFromScratch module.
    Note: class_weight is ignored in our from-scratch implementation.
    """
    print_banner("Training Custom Logistic Regression (From Scratch)")
    model = LogisticRegressionFromScratch(learning_rate=0.05, num_iterations=1000)
    model.fit(X_train, y_train)
    print("   ✅ Custom Model Converged.")
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

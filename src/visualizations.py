"""
visualizations.py
-----------------
All EDA plots, evaluation charts, and threshold analysis.

Each function saves a PNG to the reports/ directory AND returns the figure
so it can be displayed in notebooks too.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for server/script use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score,
    average_precision_score, confusion_matrix
)
import os

# ── Consistent style ─────────────────────────────────────────────────────────
PALETTE = {"legit": "#2196F3", "fraud": "#F44336"}
sns.set_style("whitegrid")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "figure.dpi": 120})

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(BASE_DIR, "static", "plots")

os.makedirs(REPORT_DIR, exist_ok=True)


def _save(fig, name: str):
    path = os.path.join(REPORT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"📊 Chart saved → {path}")
    return path


# ── 1. Class Imbalance ───────────────────────────────────────────────────────

def plot_class_distribution(y: pd.Series, title: str = "Class Distribution"):
    """Bar + donut showing severe class imbalance."""
    counts = y.value_counts().sort_index()
    labels = ["Legitimate", "Fraudulent"]
    colors = [PALETTE["legit"], PALETTE["fraud"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    # Bar chart
    bars = ax1.bar(labels, counts.values, color=colors, edgecolor="white",
                   linewidth=1.5, width=0.5)
    for bar, count in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                 f"{count:,}\n({count/len(y):.2%})",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Number of Transactions")
    ax1.set_title("Transaction Counts")
    ax1.set_ylim(0, counts.max() * 1.15)
    ax1.tick_params(axis="x", labelsize=12)

    # Donut
    wedges, texts, autotexts = ax2.pie(
        counts.values, labels=labels, colors=colors,
        autopct="%1.2f%%", startangle=90, pctdistance=0.75,
        wedgeprops={"width": 0.5, "edgecolor": "white", "linewidth": 2}
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    ax2.set_title("Proportion of Classes")

    plt.tight_layout()
    _save(fig, "01_class_distribution.png")
    plt.close()
    return fig


# ── 2. Feature Distributions ─────────────────────────────────────────────────

def plot_feature_distributions(df: pd.DataFrame, features: list):
    """KDE plots comparing fraud vs legit for key features."""
    n_cols = 3
    n_rows = (len(features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    axes = axes.flatten()
    fig.suptitle("Feature Distributions: Fraud vs Legitimate",
                 fontsize=14, fontweight="bold", y=1.01)

    for i, feat in enumerate(features):
        ax = axes[i]
        legit_vals = df[df["Class"] == 0][feat].clip(
            df[feat].quantile(0.01), df[feat].quantile(0.99))
        fraud_vals = df[df["Class"] == 1][feat].clip(
            df[feat].quantile(0.01), df[feat].quantile(0.99))

        ax.hist(legit_vals, bins=50, alpha=0.55, color=PALETTE["legit"],
                density=True, label="Legitimate")
        ax.hist(fraud_vals, bins=50, alpha=0.65, color=PALETTE["fraud"],
                density=True, label="Fraud")
        ax.set_title(feat, fontweight="bold")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    _save(fig, "02_feature_distributions.png")
    plt.close()
    return fig


# ── 3. Correlation Heatmap ───────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, cols: list):
    """Pearson correlation matrix heatmap."""
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))   # hide upper triangle

    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.3,
                annot_kws={"size": 8}, ax=ax,
                vmin=-1, vmax=1)
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "03_correlation_heatmap.png")
    plt.close()
    return fig


# ── 4. ROC Curves ────────────────────────────────────────────────────────────

def plot_roc_curves(results: list, y_test: np.ndarray):
    """
    Overlay ROC curves for multiple models.

    ROC = Receiver Operating Characteristic
    X-axis: False Positive Rate (FPR = FP / (FP+TN))
    Y-axis: True Positive Rate (TPR / Recall = TP / (TP+FN))
    AUC closer to 1.0 = better discrimination
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    for res, color in zip(results, colors):
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        auc = roc_auc_score(y_test, res["y_proba"])
        ax.plot(fpr, tpr, lw=2.5, color=color,
                label=f"{res['model']}  (AUC={auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier (AUC=0.5)")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    _save(fig, "04_roc_curves.png")
    plt.close()
    return fig


# ── 5. Precision-Recall Curves ───────────────────────────────────────────────

def plot_pr_curves(results: list, y_test: np.ndarray):
    """
    Precision-Recall curves — MORE INFORMATIVE than ROC for imbalanced data.

    Why? ROC can look great even on imbalanced data because TN is huge.
    PR curves focus only on the minority class performance.

    Random baseline = fraud prevalence (~0.17%), not 0.5.
    """
    fraud_rate = y_test.mean()
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    for res, color in zip(results, colors):
        prec, rec, _ = precision_recall_curve(y_test, res["y_proba"])
        ap = average_precision_score(y_test, res["y_proba"])
        ax.plot(rec, prec, lw=2.5, color=color,
                label=f"{res['model']}  (AP={ap:.4f})")

    ax.axhline(fraud_rate, color="gray", linestyle="--", lw=1.5,
               label=f"Random baseline ({fraud_rate:.3%})")
    ax.set_xlabel("Recall (Fraud caught / All fraud)", fontsize=12)
    ax.set_ylabel("Precision (Correct fraud alerts / All alerts)", fontsize=12)
    ax.set_title("Precision-Recall Curves — Model Comparison",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    _save(fig, "05_pr_curves.png")
    plt.close()
    return fig


# ── 6. Confusion Matrix ───────────────────────────────────────────────────────

def plot_confusion_matrix(y_test, y_pred, model_name: str, threshold: float):
    """Annotated confusion matrix with business labels."""
    cm = confusion_matrix(y_test, y_pred)
    labels = [["True Negative\n(Correctly cleared)", "False Positive\n(False alarm)"],
              ["False Negative\n(Missed fraud!)", "True Positive\n(Fraud caught)"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Legit", "Predicted Fraud"], fontsize=12)
    ax.set_yticklabels(["Actual Legit", "Actual Fraud"], fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}\n(threshold={threshold:.3f})",
                 fontsize=13, fontweight="bold")

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
            ax.text(j, i,
                    f"{cm[i, j]:,}\n{labels[i][j]}",
                    ha="center", va="center", color=color, fontsize=10,
                    fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fname = f"06_confusion_{model_name.replace(' ', '_').lower()}.png"
    _save(fig, fname)
    plt.close()
    return fig


# ── 7. Threshold Analysis ─────────────────────────────────────────────────────

def plot_threshold_analysis(y_test: np.ndarray, y_proba: np.ndarray,
                             model_name: str = "Model",
                             optimal_threshold: float = None):
    """
    Shows how Precision & Recall trade off across all thresholds.

    Business insight:
    - Bank wants HIGH RECALL → lower threshold (catch more fraud, more alarms)
    - Customer wants HIGH PRECISION → raise threshold (fewer false declines)
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()

        p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r + 1e-9)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, precisions, label="Precision", color="#2196F3", lw=2.5)
    ax.plot(thresholds, recalls,   label="Recall",    color="#F44336", lw=2.5)
    ax.plot(thresholds, f1s,       label="F1-Score",  color="#4CAF50", lw=2.5,
            linestyle="--")

    if optimal_threshold is not None:
        ax.axvline(optimal_threshold, color="#FF9800", linestyle=":", lw=2.5,
                   label=f"Optimal threshold = {optimal_threshold:.3f}")

    ax.axvline(0.5, color="gray", linestyle=":", lw=1.5,
               label="Default threshold = 0.5")

    ax.set_xlabel("Decision Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Threshold Analysis — {model_name}",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    _save(fig, "07_threshold_analysis.png")
    plt.close()
    return fig


# ── 8. Feature Importance ─────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names: list, model_name: str = "RF"):
    """Bar chart of feature importances (Random Forest)."""
    if not hasattr(model, "feature_importances_"):
        print("⚠  Model has no feature_importances_; skipping plot.")
        return None

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:15]  # top 15

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(idx)))
    bars = ax.barh(range(len(idx)),
                   importances[idx][::-1],
                   color=colors[::-1], edgecolor="white")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx[::-1]], fontsize=11)
    ax.set_xlabel("Feature Importance (Gini)", fontsize=12)
    ax.set_title(f"Top 15 Feature Importances — {model_name}",
                 fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.4)
    plt.tight_layout()
    _save(fig, "08_feature_importance.png")
    plt.close()
    return fig


# ── 9. Business Scenario Comparison ──────────────────────────────────────────

def plot_business_scenarios(y_test: np.ndarray, y_proba: np.ndarray):
    """
    Visualise 3 business scenarios:
      - Aggressive (low threshold): catch everything, lots of false alarms
      - Balanced (optimal threshold)
      - Conservative (high threshold): fewer alarms, miss some fraud
    """
    scenarios = {
        "Aggressive\n(T=0.2)":    0.2,
        "Balanced\n(T=0.4)":      0.4,
        "Conservative\n(T=0.7)":  0.7,
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Business Scenarios: Threshold Trade-off",
                 fontsize=14, fontweight="bold")

    for ax, (label, t) in zip(axes, scenarios.items()):
        y_pred = (y_proba >= t).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        total_fraud = tp + fn

        colors_cm = [["#E3F2FD", "#FFCDD2"], ["#FFCDD2", "#C8E6C9"]]
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())

        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]:,}",
                        ha="center", va="center", fontsize=13, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() * 0.6 else "black")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Legit", "Pred Fraud"])
        ax.set_yticklabels(["Actual Legit", "Actual Fraud"])
        recall_val = tp / total_fraud if total_fraud > 0 else 0
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
        ax.set_title(f"{label}\nRecall={recall_val:.1%}  Prec={precision_val:.1%}",
                     fontweight="bold", fontsize=11)

    plt.tight_layout()
    _save(fig, "09_business_scenarios.png")
    plt.close()
    return fig


# ── 10. Comparison Table Chart ────────────────────────────────────────────────

def plot_model_comparison(results: list):
    """Grouped bar chart comparing all model metrics."""
    metrics = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    labels  = ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    x = np.arange(len(metrics))
    width = 0.8 / len(results)
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (res, color) in enumerate(zip(results, colors)):
        vals = [res[m] for m in metrics]
        offset = (i - len(results) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=res["model"],
                      color=color, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.18)
    ax.set_title("Model Comparison — All Metrics", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    _save(fig, "10_model_comparison.png")
    plt.close()
    return fig

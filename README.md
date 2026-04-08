# 🛡 FraudShield — AI-Powered Fraud Detection System

> A production-inspired, end-to-end machine learning system for detecting financial transaction fraud. Built for internship portfolios and AI/ML roles.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-green?style=flat-square&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

---

## 📌 Problem Statement

Financial fraud is rare (~0.17% of transactions) but costly. A naive model that predicts "legitimate" for every transaction achieves **99.83% accuracy** — but catches **zero fraud**. This project tackles the real challenge: detecting fraud in a severely imbalanced dataset, with full control over the precision–recall trade-off.

---

## 🏗 Project Structure

```
fraud_detection/
├── data/
│   └── transactions.csv          # 100,000 synthetic transactions
├── src/
│   ├── data_generator.py         # Realistic synthetic dataset generator
│   ├── preprocessing.py          # Preprocessor class (scaling, splitting)
│   ├── imbalance_handler.py      # SMOTE, oversampling, undersampling
│   ├── models.py                 # LR, Random Forest, Isolation Forest
│   ├── visualizations.py         # All EDA & evaluation charts
│   └── train.py                  # End-to-end training pipeline ← START HERE
├── app/
│   ├── app.py                    # Flask web application
│   └── templates/
│       ├── index.html            # Single transaction prediction UI
│       ├── simulate.html         # Real-time transaction feed simulation
│       └── dashboard.html        # Model metrics & explanations
├── models/                       # Saved .joblib models
├── reports/                      # All generated charts (PNG)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/fraud-detection-system
cd fraud-detection-system/fraud_detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full training pipeline
python src/train.py

# 4. Launch the web app
python app/app.py

# 5. Open http://127.0.0.1:5000
```

---

## 📊 Results Summary

| Model | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.5 | 0.895 | **1.000** | 0.944 | 1.000 | 0.998 |
| LR + SMOTE | 0.5 | 0.971 | 0.971 | 0.971 | 1.000 | 0.999 |
| Random Forest (default) | 0.5 | **1.000** | 0.765 | 0.867 | 1.000 | 0.998 |
| **RF + Tuned Threshold ⭐** | **0.189** | **0.944** | **1.000** | **0.971** | **1.000** | **0.998** |
| Isolation Forest (unsupervised) | auto | 0.879 | 0.853 | 0.866 | 1.000 | 0.957 |

> ⭐ **Best model:** Random Forest with F2-optimised threshold — catches 100% of fraud with only 2 false alarms per 20,000 transactions.

---

## 🔑 Key Concepts Demonstrated

### 1. Why Accuracy Fails Here
With 0.17% fraud, a model predicting "all legitimate" gets 99.83% accuracy but 0% recall. We use **Precision-Recall AUC** as the primary metric.

### 2. Class Imbalance Handling
Three strategies compared:
- **`class_weight='balanced'`** — adjusts the loss function, no resampling
- **SMOTE** — creates synthetic fraud examples via KNN interpolation
- **Random Undersampling** — removes majority class examples

### 3. Threshold Tuning
Default threshold of 0.5 is rarely optimal. We maximise the **F2-score** (recall-weighted) to find the optimal cut-off. This is a core production concept — the threshold should reflect business cost.

```python
# Business trade-off:
# Lower threshold → higher recall → catch more fraud → more false alarms
# Higher threshold → higher precision → fewer false alarms → miss some fraud
```

### 4. Anomaly Detection (Unsupervised)
Isolation Forest finds fraud without labels — useful when historical fraud data is scarce. Achieved PR-AUC of 0.957 purely from the data distribution.

### 5. Business Impact Translation
Every metric translates to a business outcome:
- **False Negative (missed fraud)** → direct financial loss
- **False Positive (false alarm)** → customer friction, analyst time
- The optimal threshold is where the cost of each balances

---

## 🌐 Web Application Pages

### `/` — Single Transaction Prediction
- Input any transaction features manually
- Choose model (RF or LR) and adjust decision threshold via slider
- Get instant verdict with probability, risk level, and flagged signals

### `/simulate` — Real-Time Feed
- Streams simulated transactions at configurable speed
- Randomly injects suspicious transactions (~5%)
- Live counters: fraud rate, avg latency, total screened

### `/dashboard` — Model Metrics
- Full comparison table of all models
- Explains why accuracy is misleading
- Covers precision vs recall trade-off, SMOTE vs class weighting

---

## 📈 Generated Reports (in `reports/`)

| File | Description |
|---|---|
| `01_class_distribution.png` | Bar + donut showing severe imbalance |
| `02_feature_distributions.png` | KDE: fraud vs legit per feature |
| `03_correlation_heatmap.png` | Pearson correlation matrix |
| `04_roc_curves.png` | All models overlaid |
| `05_pr_curves.png` | Precision-Recall curves (most important!) |
| `06_confusion_*.png` | Annotated confusion matrix with business labels |
| `07_threshold_analysis.png` | Precision/Recall/F1 vs threshold sweep |
| `08_feature_importance.png` | Random Forest top-15 features |
| `09_business_scenarios.png` | 3 threshold scenarios: aggressive/balanced/conservative |
| `10_model_comparison.png` | Grouped bar chart of all metrics |

---

## 🔌 API Endpoints

```bash
# Single prediction
POST /predict
Content-Type: application/json

{
  "hour": 3,
  "amount": 3200.00,
  "merchant_category": 5,
  "distance_from_home": 1850,
  "distance_from_last_transaction": 3200,
  "ratio_to_median_purchase": 28.7,
  "online_order": 1,
  "used_pin": 0,
  "used_chip": 0,
  "foreign_transaction": 1,
  "high_risk_country": 1,
  "v1": 2.8, "v2": -2.1, "v3": 3.1, "v4": -2.4,
  "v5": 2.6, "v6": -2.9, "v7": 2.2, "v8": -2.7,
  "threshold": 0.189,
  "model": "rf"
}

# Response:
{
  "prediction": 1,
  "label": "FRAUD",
  "probability": 0.997,
  "probability_pct": 99.7,
  "risk_level": "FRAUD",
  "threshold_used": 0.189,
  "flags": ["Online order without chip authentication", "High-risk country", ...],
  "latency_ms": 4.2
}
```

---

## 🧠 Technical Decisions

| Decision | Choice | Why |
|---|---|---|
| Baseline model | Logistic Regression | Interpretable, fast, strong baseline |
| Main model | Random Forest | Non-linear, handles imbalance, feature importance |
| Imbalance | class_weight + SMOTE | Complementary strategies |
| Threshold selection | F2 maximisation | Recall more valuable than precision in fraud |
| Anomaly detection | Isolation Forest | Unsupervised, works without labels |
| Scaling | StandardScaler on train only | Prevents data leakage |
| Split | Stratified | Preserves class ratio in test set |

---

## 💼 Business Impact

A model deployed at a mid-size bank processing **10 million transactions/month** with 0.17% fraud rate (17,000 fraud cases):

| Metric | Default (0.5 threshold) | Tuned (0.189 threshold) |
|---|---|---|
| Fraud cases caught | 13,000 (76.5%) | **17,000 (100%)** |
| Fraud missed (direct loss) | 4,000 cases | **0 cases** |
| False alarms/day | ~0 | ~33 |
| Analyst reviews needed/day | ~0 | **~33** (manageable) |

The tuned model **eliminates fraud losses** at the cost of 33 analyst reviews per day — a clear business win.

---

## 🔮 Possible Extensions

- [ ] XGBoost / LightGBM for potentially higher PR-AUC
- [ ] SHAP values for per-transaction explainability
- [ ] Graph Neural Networks for card network fraud patterns
- [ ] Online learning for concept drift (fraud patterns change)
- [ ] Alert email system for flagged high-risk transactions
- [ ] Time-series features (velocity checks)
- [ ] Docker containerisation for deployment

---

## 👤 Author

Built as a portfolio project demonstrating production ML engineering skills.

**Stack:** Python · scikit-learn · Flask · Jinja2 · NumPy · Pandas · Matplotlib · Seaborn

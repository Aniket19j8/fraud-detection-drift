# 🔍 Fraud Detection with Drift-Aware Evaluation

An **adaptive ML system** featuring a recall-optimized ensemble (XGBoost, Random Forest, Logistic Regression) on 280K+ imbalanced credit card transactions, with concept drift monitoring via PSI/KS tests and adaptive threshold recalibration.

## ✨ Key Results

| Metric | Value |
|--------|-------|
| **Recall** | ~83% at 0.17% fraud rate |
| **Evaluation** | Prequential time-ordered (no leakage) |
| **Drift Recovery** | ~25% recall restoration on shifted distributions |
| **Manual Review** | < 1% of transactions flagged |
| **SMOTE vs Class-Weight** | Class weights superior for PCA features |

## 🏗️ Architecture

```
Raw Transactions
      │
      ▼
Feature Engineering (log-amount, time features, interactions)
      │
      ▼
RobustScaler → PCA (20 components)
      │
      ├──────────────────────────────┐
      ▼                              ▼
Class-Weight Ensemble           SMOTE Ensemble (comparison)
(XGB + RF + LR, soft vote)     (XGB + RF + LR, soft vote)
      │                              │
      ▼                              ▼
F2-Optimized Threshold Selection
      │
      ▼
Drift Monitor (PSI + KS two-sample tests per feature)
      │
      ▼
Adaptive Recalibrator → restored recall on drifted streams
```

## 🚀 Quick Start

### Run the Full Pipeline (CLI)
```bash
pip install -r requirements.txt

# With real Kaggle data (optional — auto-generates synthetic otherwise):
# Download https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place as data/creditcard.csv

python main.py
```

### Interactive Streamlit Dashboard
```bash
streamlit run app.py
```

### Docker
```bash
docker build -t fraud-detection .
docker run -p 8501:8501 fraud-detection
```

## 📁 Project Structure

```
├── main.py                    # Full pipeline: train → evaluate → drift → recalibrate → plot
├── app.py                     # Streamlit interactive dashboard
├── drift_monitor.py           # PSI + KS drift detection engine
├── adaptive_recalibrator.py   # Dynamic threshold adjustment
├── utils.py                   # Data loading, feature engineering, plotting
├── tests/
│   ├── test_drift_monitor.py  # Drift monitor unit tests
│   ├── test_recalibrator.py   # Recalibrator unit tests
│   └── test_pipeline.py       # End-to-end pipeline tests
├── Dockerfile
├── requirements.txt
└── README.md
```

## 🔬 How It Works

### 1. Recall-Optimized Ensemble
Three classifiers vote with soft probabilities, weighted 3:2:1 (XGB:RF:LR). Class weights penalize false negatives heavily, pushing the ensemble toward high recall.

### 2. SMOTE Comparison
A parallel SMOTE-based ensemble is trained for fair comparison. On PCA-transformed features, class-weighting consistently outperforms SMOTE in precision while maintaining comparable recall.

### 3. Drift Monitoring
Each incoming data window is checked against the training distribution using:
- **PSI (Population Stability Index)**: Measures distribution shift in binned histograms
- **KS Test**: Non-parametric two-sample test per feature
- Severity classification: LOW → MODERATE → HIGH → CRITICAL

### 4. Adaptive Recalibration
When drift is detected, the recalibrator:
1. Computes a new precision-recall curve on a recent labeled window
2. Adjusts the threshold with drift-severity-weighted sensitivity
3. Enforces constraints: manual review rate < 1%, minimum precision floor
4. Restores ~25% recall on drifted distributions

## 🧪 Testing

```bash
pytest tests/ -v
```

## License

MIT

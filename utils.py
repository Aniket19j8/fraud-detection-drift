"""
Utilities — Data loading, feature engineering, plotting
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────

def load_data(path: str = "data/creditcard.csv") -> pd.DataFrame:
    """
    Load the credit card fraud dataset.
    If not found locally, generate a synthetic version for demo.
    """
    p = Path(path)
    if p.exists():
        logger.info(f"Loading data from {p}")
        return pd.read_csv(p)
    
    logger.warning(f"Dataset not found at {p} — generating synthetic data for demo")
    return _generate_synthetic_data()


def _generate_synthetic_data(n_samples: int = 284807, fraud_rate: float = 0.00172) -> pd.DataFrame:
    """
    Generate synthetic credit card transaction data mimicking the Kaggle dataset.
    28 PCA features (V1-V28) + Time + Amount + Class.
    """
    np.random.seed(42)
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    # Legit transactions: mostly centered around 0
    legit_features = np.random.randn(n_legit, 28) * 1.0
    legit_amount = np.abs(np.random.lognormal(mean=4.0, sigma=1.5, size=n_legit))
    legit_time = np.sort(np.random.uniform(0, 172800, n_legit))

    # Fraud transactions: different distribution (shifted means)
    fraud_features = np.random.randn(n_fraud, 28) * 1.5
    # Key discriminative features
    fraud_features[:, 0] -= 2.0   # V1 shifted
    fraud_features[:, 1] += 1.5   # V2 shifted
    fraud_features[:, 3] += 2.0   # V4 shifted
    fraud_features[:, 9] -= 1.5   # V10 shifted
    fraud_features[:, 11] += 1.8  # V12 shifted
    fraud_features[:, 13] -= 1.5  # V14 shifted
    fraud_features[:, 16] -= 1.0  # V17 shifted
    fraud_amount = np.abs(np.random.lognormal(mean=3.5, sigma=2.0, size=n_fraud))
    fraud_time = np.sort(np.random.uniform(0, 172800, n_fraud))

    # Combine
    features = np.vstack([legit_features, fraud_features])
    amount = np.concatenate([legit_amount, fraud_amount])
    time_col = np.concatenate([legit_time, fraud_time])
    labels = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])

    # Shuffle
    idx = np.random.permutation(n_samples)
    features = features[idx]
    amount = amount[idx]
    time_col = np.sort(time_col)  # Keep time ordered
    labels = labels[idx]

    # Build DataFrame
    columns = [f"V{i}" for i in range(1, 29)]
    df = pd.DataFrame(features, columns=columns)
    df.insert(0, "Time", time_col)
    df["Amount"] = amount
    df["Class"] = labels.astype(int)

    return df


# ──────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to the transaction dataset."""
    df = df.copy()

    # Log-transform Amount
    df["Log_Amount"] = np.log1p(df["Amount"])

    # Time-based features (assuming Time is seconds from first transaction)
    df["Hour"] = (df["Time"] / 3600).astype(int) % 24
    df["Is_Night"] = ((df["Hour"] >= 22) | (df["Hour"] <= 5)).astype(int)

    # Amount statistics per hour
    df["Amount_zscore"] = (df["Amount"] - df["Amount"].mean()) / (df["Amount"].std() + 1e-8)

    # Interaction features
    df["V1_V2"] = df["V1"] * df["V2"]
    df["V1_V3"] = df["V1"] * df["V3"]
    df["V_high_risk_sum"] = df["V14"] + df["V12"] + df["V10"]

    # Rolling statistics (simulated via expanding window on time-sorted data)
    df = df.sort_values("Time").reset_index(drop=True)
    df["Amount_rolling_mean"] = df["Amount"].expanding(min_periods=1).mean()
    df["Amount_rolling_std"] = df["Amount"].expanding(min_periods=1).std().fillna(0)

    return df


# ──────────────────────────────────────────────────────────
# TIME-ORDERED SPLITTING
# ──────────────────────────────────────────────────────────

def create_time_ordered_splits(df: pd.DataFrame, test_size: float = 0.2):
    """
    Prequential (time-ordered) train/test split — no data leakage.
    """
    df = df.sort_values("Time").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))

    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    feature_cols = [c for c in df.columns if c not in ("Class", "Time")]
    X_train = train[feature_cols]
    y_train = train["Class"]
    X_test = test[feature_cols]
    y_test = test["Class"]

    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, title: str, save_path: Path):
    """Plot and save confusion matrix."""
    cm = np.array([[0, 0], [0, 0]])
    for t, p in zip(y_true, y_pred):
        cm[int(t)][int(p)] += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legit", "Fraud"],
        yticklabels=["Legit", "Fraud"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {title}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_precision_recall(y_true, y_prob_cw, y_prob_smote, save_path: Path):
    """Plot PR curves for both approaches."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for probs, label, color in [
        (y_prob_cw, "Class-Weight", "#2ecc71"),
        (y_prob_smote, "SMOTE", "#e74c3c"),
    ]:
        p, r, _ = precision_recall_curve(y_true, probs)
        ap = average_precision_score(y_true, probs)
        ax.plot(r, p, color=color, label=f"{label} (AP={ap:.3f})", linewidth=2)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_true, y_prob_cw, y_prob_smote, save_path: Path):
    """Plot ROC curves for both approaches."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for probs, label, color in [
        (y_prob_cw, "Class-Weight", "#2ecc71"),
        (y_prob_smote, "SMOTE", "#e74c3c"),
    ]:
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, label=f"{label} (AUC={roc_auc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(ensemble, feature_names, save_path: Path):
    """Plot feature importance from XGBoost estimator in the ensemble."""
    try:
        xgb_model = ensemble.named_estimators_["xgb"]
        importances = xgb_model.feature_importances_

        if feature_names is not None and len(feature_names) > len(importances):
            # PCA features
            names = [f"PC{i+1}" for i in range(len(importances))]
        elif feature_names is not None:
            names = list(feature_names)[:len(importances)]
        else:
            names = [f"Feature_{i}" for i in range(len(importances))]

        # Top 15
        idx = np.argsort(importances)[-15:]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(idx)), importances[idx], color="#3498db")
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([names[i] for i in idx])
        ax.set_xlabel("Importance")
        ax.set_title("Top 15 Feature Importances (XGBoost)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.warning(f"Could not plot feature importance: {e}")


def plot_drift_report(drift_results: list[dict], save_path: Path):
    """Visualize drift across monitoring windows."""
    if not drift_results:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    windows = [d["window_name"] for d in drift_results]
    psi_means = [d["psi_mean"] for d in drift_results]
    ks_means = [d["ks_mean"] for d in drift_results]
    drift_fracs = [d["drift_fraction"] for d in drift_results]

    colors = ["#2ecc71" if not d["drift_detected"] else "#e74c3c" for d in drift_results]

    axes[0].bar(windows, psi_means, color=colors)
    axes[0].axhline(y=0.2, color="red", linestyle="--", alpha=0.7, label="PSI Threshold")
    axes[0].set_title("Mean PSI per Window")
    axes[0].legend()

    axes[1].bar(windows, ks_means, color=colors)
    axes[1].set_title("Mean KS Statistic per Window")

    axes[2].bar(windows, [f * 100 for f in drift_fracs], color=colors)
    axes[2].set_title("% Features Drifted")
    axes[2].set_ylabel("Percent")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────

def save_metrics(data: dict, path: Path):
    """Save metrics to JSON."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

"""
Streamlit Dashboard — Interactive Fraud Detection Demo
======================================================
Provides interactive controls for exploring the fraud detection
pipeline, drift monitoring, and recalibration.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
)
from xgboost import XGBClassifier

from drift_monitor import DriftMonitor
from adaptive_recalibrator import AdaptiveRecalibrator
from utils import load_data, engineer_features, create_time_ordered_splits

st.set_page_config(page_title="🔍 Fraud Detection Dashboard", layout="wide")

# ──────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────

st.sidebar.title("⚙️ Configuration")

class_weight_mult = st.sidebar.slider("Class Weight Multiplier", 10, 100, 50)
pca_components = st.sidebar.slider("PCA Components", 5, 28, 20)
threshold = st.sidebar.slider("Classification Threshold", 0.01, 0.99, 0.5, 0.01)
drift_noise = st.sidebar.slider("Simulated Drift Noise (σ)", 0.0, 2.0, 0.5, 0.1)
test_size = st.sidebar.slider("Test Set Fraction", 0.1, 0.4, 0.2, 0.05)

# ──────────────────────────────────────────────────────────
# Load & Process
# ──────────────────────────────────────────────────────────

@st.cache_data
def get_data():
    df = load_data()
    df = engineer_features(df)
    return df

st.title("🔍 Drift-Aware Fraud Detection System")
st.markdown("**Recall-optimized ensemble** with PSI drift tracking and adaptive threshold recalibration.")

with st.spinner("Loading data …"):
    df = get_data()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", f"{len(df):,}")
col2.metric("Fraud Cases", f"{int(df['Class'].sum()):,}")
col3.metric("Fraud Rate", f"{df['Class'].mean():.4%}")
col4.metric("Features", f"{df.shape[1] - 1}")

# ──────────────────────────────────────────────────────────
# Train Model
# ──────────────────────────────────────────────────────────

@st.cache_resource
def train_model(_df, cw, pca_n, ts):
    X_train, X_test, y_train, y_test = create_time_ordered_splits(_df, test_size=ts)
    
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    pca = PCA(n_components=pca_n, random_state=42)
    X_train_p = pca.fit_transform(X_train_s)
    X_test_p = pca.transform(X_test_s)
    
    ensemble = VotingClassifier(
        estimators=[
            ("xgb", XGBClassifier(n_estimators=200, max_depth=6, scale_pos_weight=cw,
                                   eval_metric="aucpr", random_state=42, use_label_encoder=False)),
            ("rf", RandomForestClassifier(n_estimators=100, class_weight={0: 1, 1: cw},
                                          random_state=42, n_jobs=-1)),
            ("lr", LogisticRegression(class_weight={0: 1, 1: cw}, max_iter=1000, random_state=42)),
        ],
        voting="soft", weights=[3, 2, 1],
    )
    ensemble.fit(X_train_p, y_train)
    
    return ensemble, X_train_p, X_test_p, y_train, y_test, scaler, pca

with st.spinner("Training ensemble …"):
    model, X_train_pca, X_test_pca, y_train, y_test, scaler, pca = train_model(
        df, class_weight_mult, pca_components, test_size
    )

# ──────────────────────────────────────────────────────────
# Evaluation Tab
# ──────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📊 Model Performance", "🌊 Drift Monitor", "🔄 Recalibration"])

with tab1:
    y_prob = model.predict_proba(X_test_pca)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    recall = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred)
    review_rate = y_pred.mean()
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recall", f"{recall:.4f}")
    m2.metric("Precision", f"{prec:.4f}")
    m3.metric("F1 Score", f"{f1:.4f}")
    m4.metric("Review Rate", f"{review_rate:.4%}")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    
    with col_b:
        st.subheader("Precision-Recall Curve")
        p_vals, r_vals, t_vals = precision_recall_curve(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(r_vals, p_vals, color="#2ecc71", linewidth=2)
        ax.axvline(x=recall, color="red", linestyle="--", alpha=0.7, label=f"Current Recall={recall:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

with tab2:
    st.subheader("🌊 Concept Drift Monitoring")
    
    drift_monitor = DriftMonitor(reference_data=X_train_pca)
    n_windows = 5
    window_size = len(X_test_pca) // n_windows
    
    drift_data = []
    for i in range(n_windows):
        start = i * window_size
        end = (i + 1) * window_size
        window = X_test_pca[start:end]
        report = drift_monitor.check_drift(window, window_name=f"Window {i+1}")
        drift_data.append(report)
    
    drift_df = pd.DataFrame([{
        "Window": d["window_name"],
        "PSI Mean": d["psi_mean"],
        "KS Mean": d["ks_mean"],
        "Drift %": f"{d['drift_fraction']:.1%}",
        "Severity": d["severity"],
        "Drift?": "🔴 YES" if d["drift_detected"] else "🟢 NO",
    } for d in drift_data])
    
    st.dataframe(drift_df, use_container_width=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    windows = [d["window_name"] for d in drift_data]
    colors = ["#e74c3c" if d["drift_detected"] else "#2ecc71" for d in drift_data]
    
    axes[0].bar(windows, [d["psi_mean"] for d in drift_data], color=colors)
    axes[0].axhline(y=0.2, color="red", linestyle="--", alpha=0.7)
    axes[0].set_title("PSI per Window")
    
    axes[1].bar(windows, [d["ks_mean"] for d in drift_data], color=colors)
    axes[1].set_title("KS Statistic per Window")
    
    plt.tight_layout()
    st.pyplot(fig)

with tab3:
    st.subheader("🔄 Adaptive Recalibration")
    
    np.random.seed(42)
    X_test_drifted = X_test_pca + np.random.normal(0, drift_noise, X_test_pca.shape)
    y_prob_drifted = model.predict_proba(X_test_drifted)[:, 1]
    
    y_pred_before = (y_prob_drifted >= threshold).astype(int)
    recall_before = recall_score(y_test, y_pred_before)
    
    recalibrator = AdaptiveRecalibrator(base_threshold=threshold, sensitivity=0.3)
    new_thresh = recalibrator.recalibrate(
        y_true=y_test[:len(y_prob_drifted) // 2],
        y_prob=y_prob_drifted[:len(y_prob_drifted) // 2],
        drift_report=drift_data[-1] if drift_data else None,
    )
    
    y_pred_after = (y_prob_drifted >= new_thresh).astype(int)
    recall_after = recall_score(y_test, y_pred_after)
    
    r1, r2, r3 = st.columns(3)
    r1.metric("Original Threshold", f"{threshold:.4f}")
    r2.metric("Recalibrated Threshold", f"{new_thresh:.4f}")
    delta = recall_after - recall_before
    r3.metric("Recall Recovery", f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}")
    
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(
        ["Before Recalibration", "After Recalibration"],
        [recall_before, recall_after],
        color=["#e74c3c", "#2ecc71"],
    )
    for bar, val in zip(bars, [recall_before, recall_after]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_title("Recall Recovery via Adaptive Recalibration")
    st.pyplot(fig)
    
    st.info(f"Manual review rate after recalibration: {y_pred_after.mean():.4%}")

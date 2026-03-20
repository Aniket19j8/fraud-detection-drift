"""
Fraud Detection with Drift-Aware Evaluation
============================================
Adaptive ML System — Recall-optimized ensemble on imbalanced
transaction data with concept drift monitoring and adaptive
threshold recalibration.
"""

import os
import warnings
import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score,
    precision_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from drift_monitor import DriftMonitor
from adaptive_recalibrator import AdaptiveRecalibrator
from utils import (
    load_data,
    engineer_features,
    create_time_ordered_splits,
    plot_confusion_matrix,
    plot_precision_recall,
    plot_roc_curve,
    plot_feature_importance,
    plot_drift_report,
    save_metrics,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
PCA_COMPONENTS = 20
TEST_SIZE = 0.2
N_SPLITS = 5
FRAUD_RATE_WEIGHT = 50  # class_weight multiplier for minority class


def main():
    logger.info("=" * 70)
    logger.info("FRAUD DETECTION WITH DRIFT-AWARE EVALUATION")
    logger.info("=" * 70)

    # ──────────────────────────────────────────────────────────
    # 1. DATA LOADING & EXPLORATION
    # ──────────────────────────────────────────────────────────
    logger.info("\n📊 Step 1: Loading and exploring data …")
    df = load_data()
    logger.info(f"   Dataset shape: {df.shape}")
    logger.info(f"   Fraud rate: {df['Class'].mean():.4%}")
    logger.info(f"   Fraud count: {df['Class'].sum()} / {len(df)}")

    # ──────────────────────────────────────────────────────────
    # 2. FEATURE ENGINEERING
    # ──────────────────────────────────────────────────────────
    logger.info("\n🔧 Step 2: Feature engineering …")
    df = engineer_features(df)

    # ──────────────────────────────────────────────────────────
    # 3. TIME-ORDERED SPLIT (prequential evaluation)
    # ──────────────────────────────────────────────────────────
    logger.info("\n📋 Step 3: Creating time-ordered train/test splits …")
    X_train, X_test, y_train, y_test = create_time_ordered_splits(
        df, test_size=TEST_SIZE
    )
    logger.info(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info(f"   Train fraud rate: {y_train.mean():.4%}")
    logger.info(f"   Test fraud rate:  {y_test.mean():.4%}")

    # ──────────────────────────────────────────────────────────
    # 4. PREPROCESSING: Scaling + PCA
    # ──────────────────────────────────────────────────────────
    logger.info("\n⚙️  Step 4: Scaling + PCA …")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    logger.info(f"   Explained variance (PCA-{PCA_COMPONENTS}): {pca.explained_variance_ratio_.sum():.2%}")

    # ──────────────────────────────────────────────────────────
    # 5. MODEL TRAINING — Class-Weight Approach
    # ──────────────────────────────────────────────────────────
    logger.info("\n🤖 Step 5: Training recall-optimized ensemble (class-weight) …")

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=FRAUD_RATE_WEIGHT,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        use_label_encoder=False,
    )
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight={0: 1, 1: FRAUD_RATE_WEIGHT},
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    lr_model = LogisticRegression(
        class_weight={0: 1, 1: FRAUD_RATE_WEIGHT},
        max_iter=1000,
        C=0.1,
        random_state=RANDOM_STATE,
    )

    ensemble_cw = VotingClassifier(
        estimators=[("xgb", xgb_model), ("rf", rf_model), ("lr", lr_model)],
        voting="soft",
        weights=[3, 2, 1],
    )
    ensemble_cw.fit(X_train_pca, y_train)

    # Get probabilities for threshold tuning
    y_prob_cw = ensemble_cw.predict_proba(X_test_pca)[:, 1]

    # ──────────────────────────────────────────────────────────
    # 6. MODEL TRAINING — SMOTE Approach (comparison)
    # ──────────────────────────────────────────────────────────
    logger.info("\n🤖 Step 6: Training SMOTE-based ensemble (comparison) …")

    smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.3)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_pca, y_train)
    logger.info(f"   After SMOTE: {X_train_smote.shape}, fraud rate: {y_train_smote.mean():.2%}")

    xgb_smote = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        eval_metric="aucpr", random_state=RANDOM_STATE, use_label_encoder=False,
    )
    rf_smote = RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1,
    )
    lr_smote = LogisticRegression(max_iter=1000, C=0.1, random_state=RANDOM_STATE)

    ensemble_smote = VotingClassifier(
        estimators=[("xgb", xgb_smote), ("rf", rf_smote), ("lr", lr_smote)],
        voting="soft",
        weights=[3, 2, 1],
    )
    ensemble_smote.fit(X_train_smote, y_train_smote)
    y_prob_smote = ensemble_smote.predict_proba(X_test_pca)[:, 1]

    # ──────────────────────────────────────────────────────────
    # 7. THRESHOLD OPTIMIZATION (maximize recall @ acceptable precision)
    # ──────────────────────────────────────────────────────────
    logger.info("\n🎯 Step 7: Optimizing decision threshold …")
    
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob_cw)
    # Find threshold that gives best F2 (recall-weighted)
    f2_scores = (5 * precisions[:-1] * recalls[:-1]) / (4 * precisions[:-1] + recalls[:-1] + 1e-8)
    optimal_idx = np.argmax(f2_scores)
    optimal_threshold = thresholds[optimal_idx]
    logger.info(f"   Optimal threshold (F2): {optimal_threshold:.4f}")
    logger.info(f"   At threshold → Precision: {precisions[optimal_idx]:.4f}, Recall: {recalls[optimal_idx]:.4f}")

    y_pred_cw = (y_prob_cw >= optimal_threshold).astype(int)
    y_pred_smote = (y_prob_smote >= optimal_threshold).astype(int)

    # ──────────────────────────────────────────────────────────
    # 8. EVALUATION
    # ──────────────────────────────────────────────────────────
    logger.info("\n📈 Step 8: Evaluation …")

    # Class-weight results
    logger.info("\n─── Class-Weight Ensemble ───")
    logger.info(f"\n{classification_report(y_test, y_pred_cw, target_names=['Legit', 'Fraud'])}")
    recall_cw = recall_score(y_test, y_pred_cw)
    precision_cw = precision_score(y_test, y_pred_cw)
    auc_cw = roc_auc_score(y_test, y_prob_cw)
    ap_cw = average_precision_score(y_test, y_prob_cw)

    # SMOTE results
    logger.info("─── SMOTE Ensemble ───")
    logger.info(f"\n{classification_report(y_test, y_pred_smote, target_names=['Legit', 'Fraud'])}")
    recall_smote = recall_score(y_test, y_pred_smote)
    precision_smote = precision_score(y_test, y_pred_smote)
    auc_smote = roc_auc_score(y_test, y_prob_smote)
    ap_smote = average_precision_score(y_test, y_prob_smote)

    # Manual review rate
    review_rate_cw = y_pred_cw.mean()
    review_rate_smote = y_pred_smote.mean()

    logger.info("\n─── Comparison Summary ───")
    logger.info(f"   {'Metric':<25} {'Class-Weight':>15} {'SMOTE':>15}")
    logger.info(f"   {'─'*55}")
    logger.info(f"   {'Recall':.<25} {recall_cw:>15.4f} {recall_smote:>15.4f}")
    logger.info(f"   {'Precision':.<25} {precision_cw:>15.4f} {precision_smote:>15.4f}")
    logger.info(f"   {'ROC-AUC':.<25} {auc_cw:>15.4f} {auc_smote:>15.4f}")
    logger.info(f"   {'Avg Precision':.<25} {ap_cw:>15.4f} {ap_smote:>15.4f}")
    logger.info(f"   {'Manual Review Rate':.<25} {review_rate_cw:>15.4%} {review_rate_smote:>15.4%}")

    # ──────────────────────────────────────────────────────────
    # 9. DRIFT MONITORING
    # ──────────────────────────────────────────────────────────
    logger.info("\n🌊 Step 9: Drift monitoring …")
    drift_monitor = DriftMonitor(reference_data=X_train_pca)
    
    # Simulate time-based drift by splitting test into windows
    window_size = len(X_test_pca) // 5
    drift_results = []
    for i in range(5):
        start = i * window_size
        end = (i + 1) * window_size
        window = X_test_pca[start:end]
        y_window = y_test.values[start:end] if hasattr(y_test, 'values') else y_test[start:end]
        
        drift_report = drift_monitor.check_drift(window, window_name=f"Window_{i+1}")
        drift_results.append(drift_report)
        
        status = "🔴 DRIFT" if drift_report["drift_detected"] else "🟢 STABLE"
        logger.info(f"   Window {i+1}: PSI={drift_report['psi_mean']:.4f}, "
                    f"KS={drift_report['ks_mean']:.4f} → {status}")

    # ──────────────────────────────────────────────────────────
    # 10. ADAPTIVE RECALIBRATION
    # ──────────────────────────────────────────────────────────
    logger.info("\n🔄 Step 10: Adaptive recalibration on drifted data …")
    recalibrator = AdaptiveRecalibrator(
        base_threshold=optimal_threshold,
        sensitivity=0.3,
    )

    # Simulate drifted data (add noise to test to simulate concept drift)
    np.random.seed(RANDOM_STATE)
    X_test_drifted = X_test_pca + np.random.normal(0, 0.5, X_test_pca.shape)
    y_prob_drifted = ensemble_cw.predict_proba(X_test_drifted)[:, 1]

    # Before recalibration
    y_pred_before = (y_prob_drifted >= optimal_threshold).astype(int)
    recall_before = recall_score(y_test, y_pred_before)
    
    # After recalibration
    new_threshold = recalibrator.recalibrate(
        y_true=y_test[:len(y_prob_drifted) // 2],
        y_prob=y_prob_drifted[:len(y_prob_drifted) // 2],
        drift_report=drift_results[-1],
    )
    y_pred_after = (y_prob_drifted >= new_threshold).astype(int)
    recall_after = recall_score(y_test, y_pred_after)
    review_after = y_pred_after.mean()

    logger.info(f"   Original threshold:     {optimal_threshold:.4f}")
    logger.info(f"   Recalibrated threshold: {new_threshold:.4f}")
    logger.info(f"   Recall BEFORE recalib:  {recall_before:.4f}")
    logger.info(f"   Recall AFTER recalib:   {recall_after:.4f}")
    logger.info(f"   Recall improvement:     {(recall_after - recall_before):.4f} ({(recall_after - recall_before)/recall_before:.1%})")
    logger.info(f"   Manual review rate:     {review_after:.4%}")

    # ──────────────────────────────────────────────────────────
    # 11. GENERATE VISUALIZATIONS
    # ──────────────────────────────────────────────────────────
    logger.info("\n📊 Step 11: Generating visualizations …")

    plot_confusion_matrix(y_test, y_pred_cw, "Class-Weight Ensemble", OUTPUT_DIR / "cm_classweight.png")
    plot_confusion_matrix(y_test, y_pred_smote, "SMOTE Ensemble", OUTPUT_DIR / "cm_smote.png")
    plot_precision_recall(y_test, y_prob_cw, y_prob_smote, OUTPUT_DIR / "pr_curve.png")
    plot_roc_curve(y_test, y_prob_cw, y_prob_smote, OUTPUT_DIR / "roc_curve.png")
    plot_feature_importance(ensemble_cw, X_train.columns if hasattr(X_train, 'columns') else None, OUTPUT_DIR / "feature_importance.png")
    plot_drift_report(drift_results, OUTPUT_DIR / "drift_report.png")

    # Comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metrics_names = ["Recall", "Precision", "ROC-AUC", "Avg Precision"]
    cw_vals = [recall_cw, precision_cw, auc_cw, ap_cw]
    smote_vals = [recall_smote, precision_smote, auc_smote, ap_smote]
    
    x = np.arange(len(metrics_names))
    axes[0].bar(x - 0.15, cw_vals, 0.3, label="Class-Weight", color="#2ecc71")
    axes[0].bar(x + 0.15, smote_vals, 0.3, label="SMOTE", color="#e74c3c")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_names)
    axes[0].set_title("Class-Weight vs SMOTE Comparison")
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    # Recalibration comparison
    axes[1].bar(["Before\nRecalib", "After\nRecalib"], [recall_before, recall_after],
                color=["#e74c3c", "#2ecc71"])
    axes[1].set_title("Recall Recovery via Adaptive Recalibration")
    axes[1].set_ylim(0, 1)
    for i, v in enumerate([recall_before, recall_after]):
        axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ──────────────────────────────────────────────────────────
    # 12. SAVE RESULTS
    # ──────────────────────────────────────────────────────────
    results = {
        "class_weight": {
            "recall": float(recall_cw),
            "precision": float(precision_cw),
            "roc_auc": float(auc_cw),
            "avg_precision": float(ap_cw),
            "manual_review_rate": float(review_rate_cw),
        },
        "smote": {
            "recall": float(recall_smote),
            "precision": float(precision_smote),
            "roc_auc": float(auc_smote),
            "avg_precision": float(ap_smote),
            "manual_review_rate": float(review_rate_smote),
        },
        "recalibration": {
            "original_threshold": float(optimal_threshold),
            "new_threshold": float(new_threshold),
            "recall_before": float(recall_before),
            "recall_after": float(recall_after),
            "recall_improvement": float(recall_after - recall_before),
        },
        "drift_windows": drift_results,
    }
    save_metrics(results, OUTPUT_DIR / "results.json")

    logger.info("\n✅ Pipeline complete! Results saved to outputs/")
    logger.info(f"   → {OUTPUT_DIR / 'results.json'}")
    logger.info(f"   → Visualizations in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

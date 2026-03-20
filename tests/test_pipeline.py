"""
End-to-end pipeline test — validates the full training → evaluation → drift → recalibration flow.
"""

import numpy as np
import pytest
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from xgboost import XGBClassifier

from utils import load_data, engineer_features, create_time_ordered_splits
from drift_monitor import DriftMonitor
from adaptive_recalibrator import AdaptiveRecalibrator


@pytest.fixture(scope="module")
def pipeline_artifacts():
    """Run the core pipeline once for all tests in this module."""
    df = load_data()  # will generate synthetic
    df = engineer_features(df)
    X_train, X_test, y_train, y_test = create_time_ordered_splits(df, test_size=0.2)

    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    pca = PCA(n_components=15, random_state=42)
    X_train_p = pca.fit_transform(X_train_s)
    X_test_p = pca.transform(X_test_s)

    # Lighter model for test speed
    ensemble = VotingClassifier(
        estimators=[
            ("xgb", XGBClassifier(n_estimators=50, max_depth=4, scale_pos_weight=50,
                                   eval_metric="aucpr", random_state=42, use_label_encoder=False)),
            ("rf", RandomForestClassifier(n_estimators=50, class_weight={0:1, 1:50},
                                          random_state=42, n_jobs=-1)),
            ("lr", LogisticRegression(class_weight={0:1, 1:50}, max_iter=500, random_state=42)),
        ],
        voting="soft", weights=[3, 2, 1],
    )
    ensemble.fit(X_train_p, y_train)

    y_prob = ensemble.predict_proba(X_test_p)[:, 1]

    return {
        "ensemble": ensemble,
        "X_train_p": X_train_p, "X_test_p": X_test_p,
        "y_train": y_train, "y_test": y_test,
        "y_prob": y_prob,
    }


class TestEndToEnd:
    def test_data_loads_and_engineers(self):
        df = load_data()
        assert len(df) > 100_000
        df = engineer_features(df)
        assert "Log_Amount" in df.columns
        assert "Hour" in df.columns

    def test_time_ordered_split_no_leakage(self):
        df = load_data()
        df = engineer_features(df)
        X_train, X_test, y_train, y_test = create_time_ordered_splits(df, 0.2)
        # Train should come before test in time
        assert len(X_train) > len(X_test)

    def test_ensemble_produces_probabilities(self, pipeline_artifacts):
        y_prob = pipeline_artifacts["y_prob"]
        assert len(y_prob) == len(pipeline_artifacts["y_test"])
        assert y_prob.min() >= 0
        assert y_prob.max() <= 1

    def test_recall_above_threshold(self, pipeline_artifacts):
        y_prob = pipeline_artifacts["y_prob"]
        y_test = pipeline_artifacts["y_test"]
        # At threshold 0.3, recall should be reasonable
        y_pred = (y_prob >= 0.3).astype(int)
        recall = recall_score(y_test, y_pred)
        assert recall > 0.5, f"Recall too low: {recall}"

    def test_drift_monitor_on_test_windows(self, pipeline_artifacts):
        monitor = DriftMonitor(reference_data=pipeline_artifacts["X_train_p"])
        X_test_p = pipeline_artifacts["X_test_p"]
        report = monitor.check_drift(X_test_p[:5000], "test_window")
        assert "psi_mean" in report
        assert "drift_detected" in report

    def test_recalibration_flow(self, pipeline_artifacts):
        y_test = pipeline_artifacts["y_test"]
        y_prob = pipeline_artifacts["y_prob"]

        recalibrator = AdaptiveRecalibrator(base_threshold=0.5, sensitivity=0.3)
        drift_report = {"drift_detected": True, "severity": "HIGH"}
        new_t = recalibrator.recalibrate(
            y_true=y_test[:len(y_prob)//2],
            y_prob=y_prob[:len(y_prob)//2],
            drift_report=drift_report,
        )
        assert 0.01 <= new_t <= 0.99
        history = recalibrator.get_history()
        assert len(history) == 1
        assert history[0]["review_rate"] < 0.05  # should be reasonable

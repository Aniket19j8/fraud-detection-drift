"""
Tests for AdaptiveRecalibrator
"""

import numpy as np
import pytest
from adaptive_recalibrator import AdaptiveRecalibrator


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 1000
    y_true = np.zeros(n, dtype=int)
    y_true[:50] = 1  # 5% fraud
    y_prob = np.random.beta(2, 5, n)
    y_prob[y_true == 1] = np.random.beta(5, 2, 50)  # fraud gets higher probs
    return y_true, y_prob


@pytest.fixture
def recalibrator():
    return AdaptiveRecalibrator(base_threshold=0.5, sensitivity=0.3)


class TestRecalibrate:
    def test_returns_valid_threshold(self, recalibrator, sample_data):
        y_true, y_prob = sample_data
        new_t = recalibrator.recalibrate(y_true, y_prob)
        assert 0.01 <= new_t <= 0.99

    def test_threshold_changes_with_drift(self, recalibrator, sample_data):
        y_true, y_prob = sample_data
        drift_report = {"drift_detected": True, "severity": "HIGH"}
        new_t = recalibrator.recalibrate(y_true, y_prob, drift_report=drift_report)
        assert new_t != recalibrator.base_threshold

    def test_history_recorded(self, recalibrator, sample_data):
        y_true, y_prob = sample_data
        recalibrator.recalibrate(y_true, y_prob)
        assert len(recalibrator.get_history()) == 1
        record = recalibrator.get_history()[0]
        assert "recall_at_new" in record
        assert "precision_at_new" in record
        assert "review_rate" in record


class TestShouldRecalibrate:
    def test_no_drift_no_recalibration(self, recalibrator):
        report = {"drift_detected": False, "severity": "LOW"}
        assert not recalibrator.should_recalibrate(report)

    def test_high_drift_triggers_recalibration(self, recalibrator):
        report = {"drift_detected": True, "severity": "HIGH"}
        assert recalibrator.should_recalibrate(report)

    def test_critical_drift_triggers_recalibration(self, recalibrator):
        report = {"drift_detected": True, "severity": "CRITICAL"}
        assert recalibrator.should_recalibrate(report)

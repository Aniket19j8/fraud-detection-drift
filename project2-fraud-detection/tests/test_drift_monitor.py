"""
Tests for DriftMonitor
"""

import numpy as np
import pytest
from drift_monitor import DriftMonitor


@pytest.fixture
def reference_data():
    np.random.seed(42)
    return np.random.randn(5000, 10)


@pytest.fixture
def monitor(reference_data):
    return DriftMonitor(reference_data=reference_data)


class TestPSI:
    def test_identical_distributions_low_psi(self, monitor, reference_data):
        psi = monitor.compute_psi(reference_data[:, 0], reference_data[:, 0])
        assert psi < 0.01, "PSI should be ~0 for identical distributions"

    def test_shifted_distribution_high_psi(self, monitor, reference_data):
        shifted = reference_data[:, 0] + 3.0
        psi = monitor.compute_psi(reference_data[:, 0], shifted)
        assert psi > 0.2, "PSI should be high for shifted distribution"

    def test_psi_non_negative(self, monitor, reference_data):
        np.random.seed(99)
        other = np.random.randn(5000)
        psi = monitor.compute_psi(reference_data[:, 0], other)
        assert psi >= 0


class TestKS:
    def test_identical_distributions_not_significant(self, monitor, reference_data):
        ks = monitor.compute_ks(reference_data[:, 0], reference_data[:, 0])
        assert not ks["significant"]
        assert ks["statistic"] < 0.05

    def test_different_distributions_significant(self, monitor, reference_data):
        shifted = reference_data[:, 0] + 5.0
        ks = monitor.compute_ks(reference_data[:, 0], shifted)
        assert ks["significant"]
        assert ks["p_value"] < 0.05


class TestCheckDrift:
    def test_no_drift_on_same_distribution(self, monitor, reference_data):
        report = monitor.check_drift(reference_data, window_name="test_stable")
        assert report["severity"] == "LOW"
        assert report["psi_mean"] < 0.1

    def test_drift_detected_on_shifted_data(self, monitor, reference_data):
        shifted = reference_data + 3.0
        report = monitor.check_drift(shifted, window_name="test_shifted")
        assert report["drift_detected"]
        assert report["severity"] in ("HIGH", "CRITICAL")

    def test_report_has_required_fields(self, monitor, reference_data):
        report = monitor.check_drift(reference_data, window_name="test")
        required = [
            "window_name", "n_features_checked", "psi_mean", "psi_max",
            "ks_mean", "ks_max", "drifted_features", "drift_fraction",
            "drift_detected", "severity",
        ]
        for field in required:
            assert field in report, f"Missing field: {field}"

    def test_history_tracking(self, monitor, reference_data):
        monitor.check_drift(reference_data, "w1")
        monitor.check_drift(reference_data + 1.0, "w2")
        assert len(monitor.history) == 2


class TestTrend:
    def test_no_data_trend(self, monitor):
        trend = monitor.get_trend()
        assert trend["trend"] == "no_data"

    def test_trend_with_history(self, monitor, reference_data):
        monitor.check_drift(reference_data, "w1")
        monitor.check_drift(reference_data + 2.0, "w2")
        trend = monitor.get_trend()
        assert trend["n_windows"] == 2
        assert "psi_trend" in trend

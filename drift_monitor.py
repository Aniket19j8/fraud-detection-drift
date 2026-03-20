"""
Drift Monitor — Concept Drift Detection via KS Tests & PSI
===========================================================
Monitors feature distributions for concept drift using:
  - Population Stability Index (PSI)
  - Kolmogorov-Smirnov (KS) two-sample test
  - Jensen-Shannon divergence (optional)
"""

import numpy as np
from scipy import stats
from typing import Optional


class DriftMonitor:
    """
    Monitors for concept drift between a reference (training) distribution
    and incoming data windows.
    """

    PSI_THRESHOLD = 0.2       # PSI > 0.2 → significant drift
    KS_ALPHA = 0.05           # KS test significance level
    KS_STAT_THRESHOLD = 0.1   # KS statistic threshold for alert

    def __init__(self, reference_data: np.ndarray):
        """
        Args:
            reference_data: Training data used as baseline distribution.
        """
        self.reference = reference_data
        self.n_features = reference_data.shape[1]
        self.history: list[dict] = []

    def compute_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        n_bins: int = 20,
    ) -> float:
        """
        Compute Population Stability Index between two 1-D arrays.
        
        PSI < 0.1  → No significant shift
        PSI 0.1–0.2 → Moderate shift
        PSI > 0.2  → Significant shift
        """
        # Create bins from expected distribution
        breakpoints = np.linspace(
            min(expected.min(), actual.min()) - 1e-6,
            max(expected.max(), actual.max()) + 1e-6,
            n_bins + 1,
        )
        expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
        actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)

        # Avoid log(0) by clipping
        expected_pct = np.clip(expected_pct, 1e-6, None)
        actual_pct = np.clip(actual_pct, 1e-6, None)

        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi)

    def compute_ks(self, expected: np.ndarray, actual: np.ndarray) -> dict:
        """
        Compute KS two-sample test statistic and p-value.
        """
        statistic, p_value = stats.ks_2samp(expected, actual)
        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": p_value < self.KS_ALPHA,
        }

    def check_drift(
        self,
        incoming_data: np.ndarray,
        window_name: Optional[str] = None,
    ) -> dict:
        """
        Check all features for drift against the reference distribution.

        Returns a comprehensive drift report.
        """
        n_features = min(self.n_features, incoming_data.shape[1])
        psi_values = []
        ks_results = []
        drifted_features = []

        for i in range(n_features):
            ref_col = self.reference[:, i]
            inc_col = incoming_data[:, i]

            psi = self.compute_psi(ref_col, inc_col)
            ks = self.compute_ks(ref_col, inc_col)

            psi_values.append(psi)
            ks_results.append(ks)

            if psi > self.PSI_THRESHOLD or ks["significant"]:
                drifted_features.append({
                    "feature_idx": i,
                    "psi": psi,
                    "ks_stat": ks["statistic"],
                    "ks_pvalue": ks["p_value"],
                })

        psi_mean = float(np.mean(psi_values))
        ks_stats = [r["statistic"] for r in ks_results]
        ks_mean = float(np.mean(ks_stats))
        ks_max = float(np.max(ks_stats))
        drift_pct = len(drifted_features) / n_features

        drift_detected = (
            psi_mean > self.PSI_THRESHOLD * 0.5
            or drift_pct > 0.3
            or ks_max > self.KS_STAT_THRESHOLD * 2
        )

        report = {
            "window_name": window_name or f"window_{len(self.history)}",
            "n_features_checked": n_features,
            "psi_mean": psi_mean,
            "psi_max": float(np.max(psi_values)),
            "psi_values": [round(p, 6) for p in psi_values],
            "ks_mean": ks_mean,
            "ks_max": ks_max,
            "drifted_features": drifted_features,
            "drift_fraction": drift_pct,
            "drift_detected": drift_detected,
            "severity": self._classify_severity(psi_mean, drift_pct),
        }

        self.history.append(report)
        return report

    def _classify_severity(self, psi_mean: float, drift_fraction: float) -> str:
        if psi_mean > 0.5 or drift_fraction > 0.5:
            return "CRITICAL"
        elif psi_mean > 0.2 or drift_fraction > 0.3:
            return "HIGH"
        elif psi_mean > 0.1 or drift_fraction > 0.15:
            return "MODERATE"
        else:
            return "LOW"

    def get_trend(self) -> dict:
        """Return drift trend across all monitored windows."""
        if not self.history:
            return {"trend": "no_data"}
        psi_trend = [h["psi_mean"] for h in self.history]
        return {
            "n_windows": len(self.history),
            "psi_trend": psi_trend,
            "increasing": len(psi_trend) > 1 and psi_trend[-1] > psi_trend[0],
            "max_severity": max(
                (h["severity"] for h in self.history),
                key=lambda s: ["LOW", "MODERATE", "HIGH", "CRITICAL"].index(s),
            ),
        }

"""
Adaptive Recalibrator — Dynamic Threshold Adjustment
=====================================================
Adjusts classification thresholds based on detected drift
to restore recall on shifted distributions.
"""

import numpy as np
from sklearn.metrics import precision_recall_curve, recall_score, precision_score
from typing import Optional


class AdaptiveRecalibrator:
    """
    Dynamically adjusts the classification threshold when concept drift
    is detected, optimizing for recall while keeping manual review
    workload within acceptable bounds.
    """

    MAX_REVIEW_RATE = 0.01  # Keep manual review under 1%
    MIN_PRECISION = 0.05     # Absolute minimum precision floor

    def __init__(
        self,
        base_threshold: float = 0.5,
        sensitivity: float = 0.3,
        recall_target: float = 0.80,
    ):
        """
        Args:
            base_threshold: Default classification threshold.
            sensitivity: How aggressively to adjust (0-1).
            recall_target: Target recall to achieve.
        """
        self.base_threshold = base_threshold
        self.sensitivity = sensitivity
        self.recall_target = recall_target
        self.calibration_history: list[dict] = []

    def recalibrate(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        drift_report: Optional[dict] = None,
    ) -> float:
        """
        Compute a new threshold that maximizes recall on a recent labeled
        window, subject to the review-rate and precision constraints.

        Args:
            y_true: Ground truth labels from a recent labeled window.
            y_prob: Model probabilities for the same window.
            drift_report: Optional drift report from DriftMonitor.

        Returns:
            New optimized threshold.
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        # Compute precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

        # Determine drift-adjusted sensitivity
        drift_factor = 1.0
        if drift_report and drift_report.get("drift_detected"):
            severity_map = {"LOW": 1.0, "MODERATE": 1.2, "HIGH": 1.5, "CRITICAL": 2.0}
            drift_factor = severity_map.get(drift_report.get("severity", "LOW"), 1.0)

        effective_sensitivity = min(self.sensitivity * drift_factor, 0.9)

        # Search for best threshold meeting constraints
        best_threshold = self.base_threshold
        best_score = -1

        for i, t in enumerate(thresholds):
            r = recalls[i]
            p = precisions[i]

            # Estimate review rate at this threshold
            review_rate = (y_prob >= t).mean()

            # Skip if violates constraints
            if review_rate > self.MAX_REVIEW_RATE * (1 + effective_sensitivity):
                continue
            if p < self.MIN_PRECISION:
                continue

            # Score: weighted combination favoring recall
            score = (
                0.7 * r
                + 0.2 * p
                + 0.1 * (1 - review_rate)
            )

            if score > best_score:
                best_score = score
                best_threshold = t

        # Apply sensitivity-weighted adjustment from base
        adjustment = (self.base_threshold - best_threshold) * effective_sensitivity
        new_threshold = self.base_threshold - adjustment

        # Clamp to reasonable range
        new_threshold = np.clip(new_threshold, 0.01, 0.99)

        # Record
        y_pred_new = (y_prob >= new_threshold).astype(int)
        record = {
            "base_threshold": float(self.base_threshold),
            "new_threshold": float(new_threshold),
            "adjustment": float(adjustment),
            "drift_factor": float(drift_factor),
            "effective_sensitivity": float(effective_sensitivity),
            "recall_at_new": float(recall_score(y_true, y_pred_new)) if y_true.sum() > 0 else 0.0,
            "precision_at_new": float(precision_score(y_true, y_pred_new, zero_division=0)),
            "review_rate": float(y_pred_new.mean()),
        }
        self.calibration_history.append(record)

        return float(new_threshold)

    def get_history(self) -> list[dict]:
        return self.calibration_history

    def should_recalibrate(self, drift_report: dict) -> bool:
        """Determine if recalibration is needed based on drift report."""
        if not drift_report.get("drift_detected"):
            return False
        severity = drift_report.get("severity", "LOW")
        return severity in ("MODERATE", "HIGH", "CRITICAL")

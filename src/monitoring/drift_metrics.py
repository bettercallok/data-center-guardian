"""
drift_metrics.py — Statistical Drift Detection Engine (KS + PSI)

Maintains a sliding window of recent inference inputs. At each inference call,
computes per-feature statistical drift vs. the baseline probe distribution:

  - Kolmogorov-Smirnov (KS) test: non-parametric two-sample test measuring
    the maximum distance between empirical CDFs. p-value < 0.05 signals drift.

  - Population Stability Index (PSI): binned probability ratio test. Industry
    convention: PSI < 0.1 = stable, 0.1–0.25 = minor shift, > 0.25 = major drift.

Both metrics are computed independently per SMART feature, giving operators
a feature-level breakdown of where drift is occurring.
"""
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# PSI severity thresholds (industry standard)
PSI_STABLE = 0.1
PSI_MINOR = 0.25  # > 0.25 = major drift, trigger alert

FEATURE_NAMES = [
    "smart_5_raw",
    "smart_187_raw",
    "smart_188_raw",
    "smart_197_raw",
    "smart_198_raw",
]


@dataclass
class FeatureDriftScore:
    """Drift scores for a single feature."""
    feature: str
    ks_statistic: float
    ks_p_value: float
    psi: float
    is_drifting: bool  # True if PSI > 0.25 or KS p-value < 0.05
    severity: str  # "stable", "minor", "major"


@dataclass
class DriftReport:
    """Full drift report across all features."""
    feature_scores: list[FeatureDriftScore]
    overall_drift_detected: bool
    drifting_features: list[str]
    window_size: int


def _compute_psi(baseline: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Population Stability Index between two distributions.

    PSI = sum((P_current - P_baseline) * ln(P_current / P_baseline))

    Uses a small epsilon to avoid log(0) when a bin is empty.
    """
    epsilon = 1e-6
    # Determine bin edges from combined range
    combined_min = min(baseline.min(), current.min())
    combined_max = max(baseline.max(), current.max())

    if combined_max == combined_min:
        return 0.0  # constant feature — no drift possible

    bin_edges = np.linspace(combined_min, combined_max, n_bins + 1)

    baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
    current_counts, _ = np.histogram(current, bins=bin_edges)

    # Convert to proportions
    baseline_pct = (baseline_counts + epsilon) / (baseline_counts.sum() + epsilon * n_bins)
    current_pct = (current_counts + epsilon) / (current_counts.sum() + epsilon * n_bins)

    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return float(psi)


def _severity_label(psi: float, ks_p: float) -> str:
    if psi > PSI_MINOR or ks_p < 0.01:
        return "major"
    elif psi > PSI_STABLE or ks_p < 0.05:
        return "minor"
    return "stable"


class DriftDetector:
    """
    Maintains a sliding window of recent inference inputs and computes
    per-feature drift scores against the probe baseline distribution.
    """

    def __init__(self, window_size: int = 200):
        self.window_size = window_size
        self._window: deque = deque(maxlen=window_size)
        self._baseline: Optional[np.ndarray] = None  # shape (n_probe, n_features)
        self._is_calibrated: bool = False

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    @property
    def window_count(self) -> int:
        return len(self._window)

    def calibrate(self, probe_X: np.ndarray) -> None:
        """
        Set the baseline distribution from the LHS probe samples.

        Args:
            probe_X: np.ndarray of shape (n_probe, n_features).
        """
        self._baseline = probe_X.copy()
        self._is_calibrated = True
        logger.info(
            "DriftDetector calibrated on %d probe samples across %d features.",
            len(probe_X), probe_X.shape[1]
        )

    def observe(self, X: np.ndarray) -> None:
        """
        Add a new inference sample to the sliding window.

        Args:
            X: np.ndarray of shape (1, n_features) or (n_features,).
        """
        X = np.atleast_2d(X)
        for row in X:
            self._window.append(row.copy())

    def compute_drift(self) -> Optional[DriftReport]:
        """
        Compute per-feature KS + PSI drift scores for the current window
        vs. the probe baseline.

        Returns None if not calibrated or window is too small.
        """
        if not self._is_calibrated:
            logger.warning("DriftDetector: not yet calibrated. Skipping drift computation.")
            return None
        if len(self._window) < 30:
            # Need a minimum window to get meaningful statistics
            return None

        current_matrix = np.array(self._window)  # shape (window_size, n_features)
        feature_scores = []

        for i, feat_name in enumerate(FEATURE_NAMES):
            baseline_vals = self._baseline[:, i]
            current_vals = current_matrix[:, i]

            # KS test
            ks_stat, ks_p = stats.ks_2samp(baseline_vals, current_vals)

            # PSI
            psi = _compute_psi(baseline_vals, current_vals)

            is_drifting = psi > PSI_MINOR or ks_p < 0.05
            severity = _severity_label(psi, ks_p)

            feature_scores.append(FeatureDriftScore(
                feature=feat_name,
                ks_statistic=round(float(ks_stat), 4),
                ks_p_value=round(float(ks_p), 6),
                psi=round(psi, 4),
                is_drifting=is_drifting,
                severity=severity,
            ))

        drifting = [s.feature for s in feature_scores if s.is_drifting]

        return DriftReport(
            feature_scores=feature_scores,
            overall_drift_detected=len(drifting) > 0,
            drifting_features=drifting,
            window_size=len(self._window),
        )

"""
tests/test_drift.py — KS + PSI Drift Detection Tests

Verifies that the DriftDetector correctly identifies when SMART feature
distributions shift significantly from the baseline.
"""
import numpy as np
import pytest

from src.probing.lhs_sampler import lhs_sample
from src.monitoring.drift_metrics import DriftDetector, PSI_MINOR


class TestDriftDetector:

    def test_calibration(self):
        """DriftDetector should accept the probe baseline without error."""
        probe_X = lhs_sample(200)
        detector = DriftDetector(window_size=200)
        detector.calibrate(probe_X)
        assert detector.is_calibrated

    def test_no_drift_on_stable_data(self):
        """Feeding data from the same distribution should not trigger drift."""
        rng = np.random.RandomState(0)
        probe_X = lhs_sample(300, random_seed=42)

        detector = DriftDetector(window_size=200)
        detector.calibrate(probe_X)

        # Feed 150 samples from the same distribution
        stable_X = lhs_sample(150, random_seed=99)
        for row in stable_X:
            detector.observe(row.reshape(1, -1))

        report = detector.compute_drift()
        assert report is not None
        # Stable data should not trigger MAJOR drift on all features
        major_drifting = [s for s in report.feature_scores if s.severity == "major"]
        assert len(major_drifting) == 0, (
            f"Expected no major drift on stable data, got: {major_drifting}"
        )

    def test_drift_detected_on_injected_shift(self):
        """
        Injecting extreme SMART values (all features maxed) should trigger
        FEATURE_DRIFT with high PSI on at least one feature.
        """
        probe_X = lhs_sample(300, random_seed=42)
        detector = DriftDetector(window_size=200)
        detector.calibrate(probe_X)

        # Inject severely drifted data: smart_5 and smart_187 at max values
        rng = np.random.RandomState(1)
        drifted = np.zeros((150, 5), dtype=np.float32)
        drifted[:, 0] = 500.0   # smart_5_raw maxed
        drifted[:, 1] = 200.0   # smart_187_raw maxed
        drifted[:, 2] = rng.uniform(0, 10, 150)
        drifted[:, 3] = rng.uniform(0, 10, 150)
        drifted[:, 4] = rng.uniform(0, 10, 150)

        for row in drifted:
            detector.observe(row.reshape(1, -1))

        report = detector.compute_drift()
        assert report is not None
        assert report.overall_drift_detected, "Expected drift to be detected on injected data."
        assert "smart_5_raw" in report.drifting_features or \
               "smart_187_raw" in report.drifting_features, (
            f"Expected smart_5 or smart_187 to drift. Got: {report.drifting_features}"
        )

    def test_psi_scores_are_valid_floats(self):
        """PSI scores should be non-negative floats for all features."""
        probe_X = lhs_sample(200, random_seed=42)
        detector = DriftDetector(window_size=200)
        detector.calibrate(probe_X)

        # Add enough data to compute
        window_X = lhs_sample(100, random_seed=7)
        for row in window_X:
            detector.observe(row.reshape(1, -1))

        report = detector.compute_drift()
        assert report is not None
        for score in report.feature_scores:
            assert score.psi >= 0.0, f"PSI for {score.feature} is negative: {score.psi}"
            assert 0.0 <= score.ks_statistic <= 1.0, f"KS stat out of range: {score.ks_statistic}"

    def test_minimum_window_returns_none(self):
        """compute_drift() should return None if window has fewer than 30 samples."""
        probe_X = lhs_sample(100)
        detector = DriftDetector(window_size=200)
        detector.calibrate(probe_X)

        # Only add 10 samples — below the minimum threshold
        for i in range(10):
            detector.observe(np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]))

        result = detector.compute_drift()
        assert result is None

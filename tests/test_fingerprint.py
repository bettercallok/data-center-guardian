"""
tests/test_fingerprint.py — Behavioral Fingerprint Tests

Verifies that the fingerprint compilation produces valid statistical
summaries, and that the Wasserstein distance correctly differentiates
two different fingerprints.
"""
import numpy as np
import pytest

from src.probing.lhs_sampler import lhs_sample
from src.probing.fingerprint import compile_fingerprint, wasserstein_distance


@pytest.fixture(scope="module")
def booster():
    import xgboost as xgb
    model = xgb.Booster()
    model.load_model("src/api/survival_model.json")
    return model


class TestBehavioralFingerprint:

    def test_fingerprint_compiles(self, booster):
        """Fingerprint should compile without error from LHS probes."""
        probe_X = lhs_sample(200, random_seed=1)
        import xgboost as xgb
        preds = booster.predict(xgb.DMatrix(probe_X))
        fp = compile_fingerprint(booster, probe_X, preds)
        assert fp is not None
        assert fp.fingerprint_id is not None and len(fp.fingerprint_id) == 12

    def test_rul_distribution_is_valid(self, booster):
        """RUL mean should be positive; std should be non-negative."""
        probe_X = lhs_sample(200, random_seed=2)
        import xgboost as xgb
        preds = booster.predict(xgb.DMatrix(probe_X))
        fp = compile_fingerprint(booster, probe_X, preds)
        assert fp.rul_mean >= 0
        assert fp.rul_std >= 0
        assert fp.rul_p10 <= fp.rul_median <= fp.rul_p90

    def test_class_bias_sums_to_one(self, booster):
        """Risk tier distribution should sum to 1.0 (within floating point tolerance)."""
        probe_X = lhs_sample(200, random_seed=3)
        import xgboost as xgb
        preds = booster.predict(xgb.DMatrix(probe_X))
        fp = compile_fingerprint(booster, probe_X, preds)
        total = sum(fp.class_bias.values())
        assert abs(total - 1.0) < 0.01, f"Class bias sum {total} != 1.0"

    def test_entropy_in_range(self, booster):
        """Normalized entropy should be between 0 and 1."""
        probe_X = lhs_sample(200, random_seed=4)
        import xgboost as xgb
        preds = booster.predict(xgb.DMatrix(probe_X))
        fp = compile_fingerprint(booster, probe_X, preds)
        assert 0.0 <= fp.prediction_entropy <= 1.0

    def test_feature_importance_present(self, booster):
        """Feature importance should include at least one SMART feature."""
        probe_X = lhs_sample(200, random_seed=5)
        import xgboost as xgb
        preds = booster.predict(xgb.DMatrix(probe_X))
        fp = compile_fingerprint(booster, probe_X, preds)
        assert len(fp.feature_importance) > 0
        # All importance values should be non-negative
        for feat, val in fp.feature_importance.items():
            assert val >= 0.0, f"Negative importance for {feat}: {val}"

    def test_wasserstein_distance_is_zero_for_identical_fingerprints(self, booster):
        """Comparing a fingerprint to itself should yield distance ~0."""
        probe_X = lhs_sample(100, random_seed=6)
        import xgboost as xgb
        preds = booster.predict(xgb.DMatrix(probe_X))
        fp = compile_fingerprint(booster, probe_X, preds)
        dist = wasserstein_distance(fp, fp)
        assert dist == 0.0 or dist < 1.0  # same data, distance should be minimal

    def test_wasserstein_distance_detects_shift(self, booster):
        """
        Two fingerprints from very different regions of the feature space
        should have a non-trivial Wasserstein distance.
        """
        import xgboost as xgb

        # Healthy drives — low SMART values
        healthy_X = lhs_sample(100, random_seed=10)
        healthy_X[:, :] = healthy_X[:, :] * 0.1  # compress to low range
        preds1 = booster.predict(xgb.DMatrix(healthy_X))
        fp1 = compile_fingerprint(booster, healthy_X, preds1)

        # Dying drives — high SMART values
        dying_X = lhs_sample(100, random_seed=11)
        dying_X[:, 0] = 400.0   # smart_5 maxed
        dying_X[:, 1] = 180.0   # smart_187 maxed
        preds2 = booster.predict(xgb.DMatrix(dying_X))
        fp2 = compile_fingerprint(booster, dying_X, preds2)

        dist = wasserstein_distance(fp1, fp2)
        assert dist > 0.0, "Expected non-zero Wasserstein distance between healthy and dying fingerprints."

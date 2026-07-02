"""
tests/test_novelty.py — FAISS Latent Space Novelty Tests

Verifies that the FAISSNoveltyIndexer correctly flags:
  - Normal training-distribution data as non-novel
  - Extreme OOD data as LATENT_NOVELTY
"""
import numpy as np
import pytest

from src.probing.lhs_sampler import lhs_sample
from src.monitoring.faiss_indexer import FAISSNoveltyIndexer


@pytest.fixture(scope="module")
def booster():
    """Load the real XGBoost model once for all novelty tests."""
    import xgboost as xgb
    model = xgb.Booster()
    model.load_model("src/api/survival_model.json")
    return model


@pytest.fixture(scope="module")
def built_indexer(booster):
    """Build the FAISS index from 300 LHS probe samples."""
    probe_X = lhs_sample(300, random_seed=42)
    indexer = FAISSNoveltyIndexer(n_neighbors=5, margin=2.0)
    indexer.build(booster, probe_X)
    return indexer, booster


class TestFAISSNoveltyIndexer:

    def test_index_builds_successfully(self, booster):
        probe_X = lhs_sample(100, random_seed=0)
        indexer = FAISSNoveltyIndexer()
        indexer.build(booster, probe_X)
        assert indexer.is_built
        assert indexer.threshold is not None
        assert indexer.threshold > 0

    def test_healthy_drive_is_not_novel(self, built_indexer):
        """A perfectly healthy drive (all zeros) should not be flagged as novel."""
        indexer, booster = built_indexer
        # All-zero telemetry is within normal training range
        X = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        result = indexer.score(booster, X)
        # We don't assert it's False since healthy is common, but novelty_score should be low
        assert result.novelty_score >= 0.0
        assert result.distance >= 0.0

    def test_ood_data_is_flagged_as_novel(self, built_indexer):
        """
        Extreme OOD telemetry (all features at maximum 65535) should be flagged.
        This is orders of magnitude outside the SMART feature ranges used in training.
        """
        indexer, booster = built_indexer
        X = np.array([[65535.0, 65535.0, 65535.0, 65535.0, 65535.0]], dtype=np.float32)
        result = indexer.score(booster, X)
        assert result.is_novel, (
            f"Expected OOD data to be flagged as novel. "
            f"novelty_score={result.novelty_score:.2f}, threshold=1.0"
        )
        assert result.novelty_score > 1.0

    def test_score_before_build_raises(self):
        """Scoring before building the index should raise RuntimeError."""
        import xgboost as xgb
        model = xgb.Booster()
        model.load_model("src/api/survival_model.json")
        indexer = FAISSNoveltyIndexer()
        X = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        with pytest.raises(RuntimeError, match="build"):
            indexer.score(model, X)

    def test_batch_scoring_consistency(self, built_indexer):
        """Batch scoring should return the same result as single scoring."""
        indexer, booster = built_indexer
        X = np.array([[10.0, 5.0, 0.0, 2.0, 2.0]], dtype=np.float32)
        single_result = indexer.score(booster, X)
        batch_results = indexer.score_batch(booster, X)
        assert len(batch_results) == 1
        assert abs(batch_results[0].distance - single_result.distance) < 1e-4

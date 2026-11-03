"""
faiss_indexer.py — FAISS-indexed Latent Space Novelty Monitor

At probe time, we run the XGBoost model in "predict leaf" mode, which returns
the decision tree leaf each sample ends up in. This leaf path is the model's
learned "latent fingerprint" of that data point.

Leaf IDs are categorical labels, not magnitudes, so they are one-hot encoded
(one column per leaf in the ensemble) before indexing. In that space the
squared L2 distance between two points equals 2 x the number of trees in which
they land in different leaves, i.e. FAISS computes a Hamming distance over
leaf assignments. We report it normalised to [0, 1]: the fraction of trees in
which the point disagrees with its neighbours.

At inference time, we encode the new point's leaf path and query the FAISS
index for its k-nearest neighbors. If the mean distance exceeds the 99th
percentile of k-NN distances seen among probes (times a safety margin), the
point is in uncharted territory and we flag it as LATENT_NOVELTY. The
percentile (rather than the max) keeps the threshold stable when a single
isolated probe sample happens to be far from all others.

Range guard: tree ensembles saturate beyond their outermost split thresholds,
so a point at smart_5_raw=65535 lands in the *same* leaves as 500. Leaf-space
distance alone therefore cannot see extreme extrapolation. We complement it
with a feature-space range check against the probe envelope: any feature that
exceeds `range_margin` x the probed maximum is flagged as out-of-range, and the
point is marked novel.
"""
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

import faiss

logger = logging.getLogger(__name__)


@dataclass
class NoveltyResult:
    """Result returned by a single novelty check."""
    is_novel: bool
    distance: float
    threshold: float
    novelty_score: float  # normalized 0.0–1.0+ (>1.0 means beyond threshold)
    out_of_range_features: list[str] = field(default_factory=list)


class FAISSNoveltyIndexer:
    """
    Builds and queries a FAISS flat L2 index over XGBoost leaf-path vectors.

    Leaf paths encode the model's internal representation of a data point:
    each tree votes on which leaf the point belongs to, creating a high-
    dimensional categorical fingerprint. L2 distance in this space naturally
    captures behavioral novelty.
    """

    def __init__(self, n_neighbors: int = 5, margin: float = 1.5, range_margin: float = 1.5,
                 feature_names: Optional[list[str]] = None):
        self.n_neighbors = n_neighbors
        self.margin = margin
        self.range_margin = range_margin
        self.calibration_percentile = 99
        self.feature_names = feature_names or [
            "smart_5_raw", "smart_187_raw", "smart_188_raw", "smart_197_raw", "smart_198_raw",
        ]
        self._feature_max: Optional[np.ndarray] = None
        self._leaf_lookup: Optional[np.ndarray] = None  # (n_trees, max_node+1) -> column
        self._n_trees: int = 0
        self._dim: int = 0
        self._index: Optional[faiss.IndexFlatL2] = None
        self._threshold: Optional[float] = None
        self._probe_leaf_matrix: Optional[np.ndarray] = None
        self._is_built: bool = False

    @property
    def is_built(self) -> bool:
        return self._is_built

    @property
    def threshold(self) -> Optional[float]:
        return self._threshold

    def build(self, model, probe_X: np.ndarray) -> None:
        """
        Index the probe samples' leaf paths.

        Args:
            model: An xgb.Booster with a trained survival:aft model.
            probe_X: np.ndarray of shape (n_probe, n_features) — the LHS probe samples.
        """
        import xgboost as xgb

        logger.info("FAISSNoveltyIndexer: extracting leaf paths from %d probe samples...", len(probe_X))
        self._build_leaf_lookup(model)

        encoded = self._encode(model, probe_X)
        self._probe_leaf_matrix = encoded
        self._feature_max = np.asarray(probe_X, dtype=np.float32).max(axis=0)

        self._index = faiss.IndexFlatL2(self._dim)
        self._index.add(encoded)

        # Calibrate threshold from each probe's distance to its own k nearest
        # neighbours (excluding itself, which is always at distance 0).
        k = min(self.n_neighbors + 1, len(encoded))
        distances, _ = self._index.search(encoded, k)
        mean_distances = self._normalise(distances[:, 1:]).mean(axis=1)

        base_dist = float(np.percentile(mean_distances, self.calibration_percentile))
        self._threshold = base_dist * self.margin
        self._is_built = True

        logger.info(
            "FAISSNoveltyIndexer: index built. trees=%d, leaves=%d, n_probe=%d, "
            "threshold(p%d*%.1f)=%.4f",
            self._n_trees, self._dim, len(encoded), self.calibration_percentile, self.margin, self._threshold
        )

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _build_leaf_lookup(self, model) -> None:
        """Assign every leaf in the ensemble its own one-hot column."""
        df = model.trees_to_dataframe()
        leaves = df[df["Feature"] == "Leaf"][["Tree", "Node"]].to_numpy(dtype=np.int64)
        self._n_trees = int(df["Tree"].max()) + 1
        max_node = int(df["Node"].max())
        lookup = np.full((self._n_trees, max_node + 1), -1, dtype=np.int64)
        lookup[leaves[:, 0], leaves[:, 1]] = np.arange(len(leaves))
        self._leaf_lookup = lookup
        self._dim = len(leaves)

    def _encode(self, model, X: np.ndarray) -> np.ndarray:
        import xgboost as xgb

        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        leaf_ids = np.atleast_2d(model.predict(xgb.DMatrix(X), pred_leaf=True)).astype(np.int64)
        cols = self._leaf_lookup[np.arange(self._n_trees)[None, :], leaf_ids]
        encoded = np.zeros((len(X), self._dim), dtype=np.float32)
        rows = np.repeat(np.arange(len(X)), self._n_trees)
        encoded[rows, cols.ravel()] = 1.0
        return encoded

    def _normalise(self, squared_l2: np.ndarray) -> np.ndarray:
        """Squared L2 between one-hot leaf vectors → fraction of disagreeing trees."""
        return squared_l2 / (2.0 * self._n_trees)

    def score(self, model, X: np.ndarray) -> NoveltyResult:
        """
        Score a single inference sample for latent novelty.

        Args:
            model: An xgb.Booster.
            X: np.ndarray of shape (1, n_features).

        Returns:
            NoveltyResult with distance, threshold, and novelty flag.
        """
        return self.score_batch(model, X)[0]

    def score_batch(self, model, X: np.ndarray) -> list[NoveltyResult]:
        """Score a batch of inference samples (shape (n, n_features))."""
        if not self._is_built:
            raise RuntimeError("FAISSNoveltyIndexer.build() must be called before scoring.")

        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        encoded = self._encode(model, X)

        k = min(self.n_neighbors, self._index.ntotal)
        distances, _ = self._index.search(encoded, k)
        mean_distances = self._normalise(distances).mean(axis=1)

        # Feature-space range ratio: >1.0 means beyond range_margin x probed max
        limits = np.maximum(self._feature_max, 1.0) * self.range_margin
        range_ratios = X / limits

        results = []
        for row_idx, dist in enumerate(mean_distances):
            dist = float(dist)
            leaf_score = dist / self._threshold if self._threshold > 0 else 0.0
            ratios = range_ratios[row_idx]
            oor = [self.feature_names[i] for i, r in enumerate(ratios) if r > 1.0]
            novelty_score = max(leaf_score, float(ratios.max())) if oor else leaf_score
            results.append(NoveltyResult(
                is_novel=dist > self._threshold or bool(oor),
                distance=dist,
                threshold=self._threshold,
                novelty_score=round(novelty_score, 4),
                out_of_range_features=oor,
            ))
        return results

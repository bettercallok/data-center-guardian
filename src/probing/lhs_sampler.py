"""
lhs_sampler.py — Latin Hypercube Sampling across the SMART feature space

Latin Hypercube Sampling (LHS) generates a near-uniform coverage of a
multi-dimensional feature space using far fewer samples than a full grid.
It works by dividing each feature's range into N equal-probability strata
and drawing exactly one sample from each stratum, then shuffling the
assignments across features to ensure joint coverage.

For our 5-feature SMART space, we generate N probe samples that cover the
entire plausible operating range. These samples become the "behavioral probe"
used to:
  1. Build the FAISS novelty index (leaf-path fingerprints)
  2. Calibrate the statistical drift detector (baseline distribution)
  3. Generate the behavioral fingerprint (confidence histogram, entropy)
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


# Plausible SMART feature ranges derived from the Backblaze dataset statistics.
# Ranges are [min, max] for each feature. We use the 99th percentile as max
# to avoid extreme tail values dominating the probe space.
SMART_FEATURE_RANGES = {
    "smart_5_raw":   (0, 500),    # Reallocated Sectors: healthy=0, critical=500+
    "smart_187_raw": (0, 200),    # Uncorrectable Errors: healthy=0, critical=200
    "smart_188_raw": (0, 1000),   # Command Timeouts: healthy=0
    "smart_197_raw": (0, 500),    # Current Pending Sectors: healthy=0
    "smart_198_raw": (0, 500),    # Uncorrectable Sector Count: healthy=0
}

FEATURE_NAMES = list(SMART_FEATURE_RANGES.keys())


def lhs_sample(n_samples: int, random_seed: int = 42) -> np.ndarray:
    """
    Generate Latin Hypercube Samples across the SMART feature space.

    Args:
        n_samples: Number of probe samples to generate.
        random_seed: Seed for reproducibility.

    Returns:
        np.ndarray of shape (n_samples, n_features) with values scaled to
        the SMART feature ranges.
    """
    rng = np.random.RandomState(random_seed)
    n_features = len(FEATURE_NAMES)

    # Step 1: Generate unit LHS in [0, 1]
    # Each column gets exactly one sample per stratum
    unit_lhs = np.zeros((n_samples, n_features))
    for col in range(n_features):
        # Create equally spaced strata
        strata = (rng.permutation(n_samples) + rng.uniform(size=n_samples)) / n_samples
        unit_lhs[:, col] = strata

    # Step 2: Scale to the actual feature ranges
    scaled = np.zeros_like(unit_lhs)
    for i, (feat_name, (feat_min, feat_max)) in enumerate(SMART_FEATURE_RANGES.items()):
        scaled[:, i] = unit_lhs[:, i] * (feat_max - feat_min) + feat_min

    # Step 3: Clip to integer values (SMART attributes are discrete counts)
    scaled = np.clip(scaled, 0, None).astype(np.float32)

    logger.info(
        "LHS sampler: generated %d probe samples across %d features.",
        n_samples, n_features
    )
    return scaled


def get_feature_stats(probe_X: np.ndarray) -> dict:
    """
    Return descriptive statistics for the probe distribution.
    Useful for logging and the /api/models/topology response.
    """
    stats = {}
    for i, name in enumerate(FEATURE_NAMES):
        col = probe_X[:, i]
        stats[name] = {
            "min": float(col.min()),
            "max": float(col.max()),
            "mean": float(col.mean()),
            "std": float(col.std()),
            "p25": float(np.percentile(col, 25)),
            "p50": float(np.percentile(col, 50)),
            "p75": float(np.percentile(col, 75)),
        }
    return stats

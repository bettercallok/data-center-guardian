"""
fingerprint.py — Behavioral Fingerprint Compilation

After running LHS probes through the XGBoost survival model, we compile
a statistical "behavioral fingerprint" that captures the model's response
characteristics across the full feature space:

  - RUL confidence histogram: distribution of predicted RUL days
  - Survival entropy: how "uncertain" the model is (spread of predictions)
  - Uncertainty regions: feature combinations that produce high-variance outputs
  - Class bias: proportion of predictions falling in each risk tier

The fingerprint serves as the model's behavioral identity. Future fingerprints
can be compared using Wasserstein distance to detect baseline shift after retraining.
"""
import logging
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# Risk tier thresholds (days) — must match main.py predict logic
RISK_TIERS = {
    "low": (1000, float("inf")),
    "medium": (365, 1000),
    "high": (90, 365),
    "critical": (0, 90),
}


@dataclass
class BehavioralFingerprint:
    """The compiled behavioral fingerprint of the survival model."""
    n_probe_samples: int

    # RUL distribution
    rul_mean: float
    rul_std: float
    rul_median: float
    rul_p10: float
    rul_p90: float

    # Entropy: higher = model is more "spread out" in its predictions
    # Computed as normalized entropy of the RUL histogram
    prediction_entropy: float

    # Uncertainty rate: fraction of probes where model predicts < 90 days
    uncertainty_rate: float

    # Risk tier distribution
    class_bias: dict[str, float]

    # Raw histogram for frontend charting
    histogram_bins: list[float]
    histogram_counts: list[int]

    # Feature importance extracted from XGBoost
    feature_importance: dict[str, float]

    # Fingerprint hash for versioning comparisons
    fingerprint_id: str


def compile_fingerprint(model, probe_X: np.ndarray, probe_ttf_preds: np.ndarray) -> BehavioralFingerprint:
    """
    Compile the behavioral fingerprint from LHS probe predictions.

    Args:
        model: An xgb.Booster.
        probe_X: np.ndarray of shape (n_probe, n_features).
        probe_ttf_preds: np.ndarray of shape (n_probe,) — raw TTF predictions from model.

    Returns:
        BehavioralFingerprint dataclass.
    """
    import hashlib, json

    n = len(probe_ttf_preds)
    rul_preds = np.maximum(0, probe_ttf_preds - 30)  # same formula as main.py

    # Basic statistics
    rul_mean = float(rul_preds.mean())
    rul_std = float(rul_preds.std())
    rul_median = float(np.median(rul_preds))
    rul_p10 = float(np.percentile(rul_preds, 10))
    rul_p90 = float(np.percentile(rul_preds, 90))

    # Histogram (50 bins, clipped to 3000 days for readability)
    clipped = np.clip(probe_ttf_preds, 0, 3000)
    counts, bin_edges = np.histogram(clipped, bins=50)
    histogram_bins = [round(float(e), 1) for e in bin_edges[:-1]]
    histogram_counts = [int(c) for c in counts]

    # Normalized entropy of the histogram
    probs = counts / counts.sum() if counts.sum() > 0 else counts
    probs = probs[probs > 0]
    entropy_raw = -np.sum(probs * np.log(probs))
    max_entropy = math.log(len(counts))
    prediction_entropy = float(entropy_raw / max_entropy) if max_entropy > 0 else 0.0

    # Uncertainty rate: fraction predicting < 90 days (critical)
    uncertainty_rate = float((probe_ttf_preds < 90).mean())

    # Risk tier class bias
    class_bias = {}
    for tier, (low, high) in RISK_TIERS.items():
        mask = (probe_ttf_preds >= low) & (probe_ttf_preds < high)
        class_bias[tier] = round(float(mask.mean()), 4)

    # Feature importance from the model
    importance_raw = model.get_score(importance_type="gain")
    feature_names_map = {
        "f0": "smart_5_raw",
        "f1": "smart_187_raw",
        "f2": "smart_188_raw",
        "f3": "smart_197_raw",
        "f4": "smart_198_raw",
    }
    total_gain = sum(importance_raw.values()) or 1.0
    feature_importance = {
        feature_names_map.get(k, k): round(v / total_gain, 4)
        for k, v in importance_raw.items()
    }

    # Create a deterministic fingerprint ID from core statistics
    fp_data = {
        "rul_mean": round(rul_mean, 2),
        "rul_std": round(rul_std, 2),
        "uncertainty_rate": round(uncertainty_rate, 4),
        "entropy": round(prediction_entropy, 4),
    }
    fp_hash = hashlib.sha256(json.dumps(fp_data, sort_keys=True).encode()).hexdigest()[:12]

    logger.info(
        "Fingerprint compiled: mean_rul=%.1f entropy=%.3f uncertainty_rate=%.3f id=%s",
        rul_mean, prediction_entropy, uncertainty_rate, fp_hash
    )

    return BehavioralFingerprint(
        n_probe_samples=n,
        rul_mean=round(rul_mean, 2),
        rul_std=round(rul_std, 2),
        rul_median=round(rul_median, 2),
        rul_p10=round(rul_p10, 2),
        rul_p90=round(rul_p90, 2),
        prediction_entropy=round(prediction_entropy, 4),
        uncertainty_rate=round(uncertainty_rate, 4),
        class_bias=class_bias,
        histogram_bins=histogram_bins,
        histogram_counts=histogram_counts,
        feature_importance=feature_importance,
        fingerprint_id=fp_hash,
    )


def wasserstein_distance(fp1: BehavioralFingerprint, fp2: BehavioralFingerprint) -> float:
    """
    Compute the 1D Wasserstein (Earth Mover's) distance between two
    behavioral fingerprints' RUL distributions.

    A large distance indicates the retrained model behaves differently
    from the production baseline — a signal to trigger a shadow deployment
    comparison before promoting the new model.
    """
    from scipy.stats import wasserstein_distance as wd

    # Reconstruct empirical distributions from histograms
    bins1 = np.array(fp1.histogram_bins)
    counts1 = np.array(fp1.histogram_counts)
    bins2 = np.array(fp2.histogram_bins)
    counts2 = np.array(fp2.histogram_counts)

    # Expand into value arrays for wasserstein calculation
    dist1 = np.repeat(bins1, counts1)
    dist2 = np.repeat(bins2, counts2)

    if len(dist1) == 0 or len(dist2) == 0:
        return 0.0

    distance = float(wd(dist1, dist2))
    logger.info("Wasserstein distance between fingerprints: %.4f", distance)
    return round(distance, 4)

"""
api/routes_autopsy.py — Model Autopsy Routes

POST /api/models/probe       — Run LHS probe sweep + compile behavioral fingerprint (admin)
GET  /api/models/fingerprint — Return the current behavioral fingerprint
GET  /api/models/topology    — Return extracted XGBoost tree architecture info
"""
import logging
import threading

import xgboost as xgb
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from src.api.config import settings
from src.api.security import limiter, require_admin
from src.probing.fingerprint import compile_fingerprint, wasserstein_distance
from src.probing.lhs_sampler import lhs_sample, novelty_probe_sample

FEATURE_MAP = {
    "f0": "smart_5_raw",
    "f1": "smart_187_raw",
    "f2": "smart_188_raw",
    "f3": "smart_197_raw",
    "f4": "smart_198_raw",
}

# Only one probe sweep may run at a time (it rebuilds shared monitoring state).
_probe_lock = threading.Lock()


def run_probe_sweep(state, n_samples: int):
    """Generate LHS probes and rebuild fingerprint, FAISS index and drift baseline."""
    probe_X = lhs_sample(n_samples)
    probe_ttf_preds = state.model.predict(xgb.DMatrix(probe_X))
    fp = compile_fingerprint(state.model, probe_X, probe_ttf_preds)
    # The novelty index also covers the realistic, zero-inflated healthy region.
    state.faiss_indexer.build(state.model, novelty_probe_sample(n_samples))
    state.drift_detector.calibrate(probe_X)
    previous = state.fingerprint
    state.fingerprint = fp
    return fp, previous

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/models", tags=["autopsy"])


@router.post("/probe", dependencies=[Depends(require_admin)])
@limiter.limit(settings.rate_limit_probe)
def run_probe(request: Request, n_samples: int = Query(500, ge=100, le=2000)):
    """
    Run a full LHS probe sweep (requires the admin API key):
      1. Generate n_samples LHS points across the SMART feature space
      2. Run all through the survival model
      3. Compile the behavioral fingerprint
      4. Build the FAISS novelty index
      5. Calibrate the drift detector

    It is automatically run on startup. The response includes the Wasserstein
    distance to the previous fingerprint so operators can see baseline shift.
    """
    state = request.app.state

    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if not _probe_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A probe sweep is already running.")
    try:
        logger.info("Probe sweep: generating %d LHS samples...", n_samples)
        fp, previous = run_probe_sweep(state, n_samples)
    finally:
        _probe_lock.release()

    logger.info("Probe sweep complete. Fingerprint ID: %s", fp.fingerprint_id)

    return {
        "status": "success",
        "n_samples": n_samples,
        "fingerprint_id": fp.fingerprint_id,
        "previous_fingerprint_id": previous.fingerprint_id if previous else None,
        "wasserstein_to_previous": wasserstein_distance(previous, fp) if previous else None,
        "faiss_built": state.faiss_indexer.is_built,
        "novelty_threshold": state.faiss_indexer.threshold,
        "summary": {
            "rul_mean": fp.rul_mean,
            "rul_std": fp.rul_std,
            "uncertainty_rate": fp.uncertainty_rate,
            "prediction_entropy": fp.prediction_entropy,
            "class_bias": fp.class_bias,
        },
    }


@router.get("/fingerprint")
@limiter.limit(settings.rate_limit_read)
def get_fingerprint(request: Request):
    """Return the current behavioral fingerprint of the production model."""
    state = request.app.state
    fp = state.fingerprint

    if fp is None:
        raise HTTPException(
            status_code=404,
            detail="No fingerprint compiled yet. POST /api/models/probe first."
        )

    return {
        "fingerprint_id": fp.fingerprint_id,
        "n_probe_samples": fp.n_probe_samples,
        "rul_distribution": {
            "mean": fp.rul_mean,
            "std": fp.rul_std,
            "median": fp.rul_median,
            "p10": fp.rul_p10,
            "p90": fp.rul_p90,
        },
        "prediction_entropy": fp.prediction_entropy,
        "uncertainty_rate": fp.uncertainty_rate,
        "class_bias": fp.class_bias,
        "feature_importance": fp.feature_importance,
        "histogram": {
            "bins": fp.histogram_bins,
            "counts": fp.histogram_counts,
        },
    }


@router.get("/topology")
@limiter.limit(settings.rate_limit_read)
def get_topology(request: Request):
    """
    Extract and return the XGBoost model's internal architecture.

    Returns:
      - Number of trees in the ensemble
      - Max depth of any tree
      - Feature importance (gain, weight, cover) for all 5 SMART features
      - Total number of leaves
      - Decision structure summary per tree (first 5 trees for readability)
    """
    state = request.app.state

    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    cached = getattr(state, "topology_cache", None)
    if cached is not None:
        return cached

    model = state.model

    # Extract trees as a dataframe (expensive — result is cached per model load)
    trees_df = model.trees_to_dataframe()

    n_trees = trees_df["Tree"].nunique()
    total_nodes = len(trees_df)
    total_leaves = len(trees_df[trees_df["Feature"] == "Leaf"])
    max_depth = int(trees_df["Depth"].max()) if "Depth" in trees_df.columns else None

    # Feature importance by type
    importance_types = ["gain", "weight", "cover"]
    feature_importance = {}
    feat_map = FEATURE_MAP
    for imp_type in importance_types:
        raw = model.get_score(importance_type=imp_type)
        total = sum(raw.values()) or 1.0
        feature_importance[imp_type] = {
            feat_map.get(k, k): round(v / total, 4) for k, v in raw.items()
        }

    # Summarize first 5 trees (node count, leaf count, depth)
    tree_summaries = []
    for tree_id in sorted(trees_df["Tree"].unique())[:5]:
        tree_nodes = trees_df[trees_df["Tree"] == tree_id]
        tree_leaves = tree_nodes[tree_nodes["Feature"] == "Leaf"]
        tree_depth = int(tree_nodes["Depth"].max()) if "Depth" in tree_nodes.columns else None
        tree_summaries.append({
            "tree_id": int(tree_id),
            "n_nodes": len(tree_nodes),
            "n_leaves": len(tree_leaves),
            "max_depth": tree_depth,
        })

    topology = {
        "model_type": "XGBoost Accelerated Failure Time (survival:aft)",
        "n_trees": n_trees,
        "total_nodes": total_nodes,
        "total_leaves": total_leaves,
        "max_depth": max_depth,
        "objective": "survival:aft",
        "aft_distribution": "normal",
        "feature_importance": feature_importance,
        "tree_summaries": tree_summaries,
    }
    state.topology_cache = topology
    return topology

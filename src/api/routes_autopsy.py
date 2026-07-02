"""
api/routes_autopsy.py — Model Autopsy Routes

POST /api/models/probe       — Run LHS probe sweep + compile behavioral fingerprint
GET  /api/models/fingerprint — Return the current behavioral fingerprint
GET  /api/models/topology    — Return extracted XGBoost tree architecture info
"""
import logging
import numpy as np
import xgboost as xgb

from fastapi import APIRouter, HTTPException, Request

from src.probing.lhs_sampler import lhs_sample, get_feature_stats
from src.probing.fingerprint import compile_fingerprint

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/models", tags=["autopsy"])


@router.post("/probe")
async def run_probe(request: Request, n_samples: int = 500):
    """
    Run a full LHS probe sweep:
      1. Generate n_samples LHS points across the SMART feature space
      2. Run all through the survival model
      3. Compile the behavioral fingerprint
      4. Build the FAISS novelty index
      5. Calibrate the drift detector

    This endpoint must be called at least once before novelty scoring works.
    It is automatically called on startup.
    """
    state = request.app.state

    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    n_samples = max(100, min(n_samples, 2000))  # clamp to safe range

    logger.info("Probe sweep: generating %d LHS samples...", n_samples)
    probe_X = lhs_sample(n_samples)

    # Run inference over all probe samples
    dmatrix = xgb.DMatrix(probe_X)
    probe_ttf_preds = state.model.predict(dmatrix)

    # Compile fingerprint
    fp = compile_fingerprint(state.model, probe_X, probe_ttf_preds)
    state.fingerprint = fp

    # Build FAISS index
    state.faiss_indexer.build(state.model, probe_X)

    # Calibrate drift detector
    state.drift_detector.calibrate(probe_X)

    logger.info("Probe sweep complete. Fingerprint ID: %s", fp.fingerprint_id)

    return {
        "status": "success",
        "n_samples": n_samples,
        "fingerprint_id": fp.fingerprint_id,
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
async def get_fingerprint(request: Request):
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
async def get_topology(request: Request):
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

    model = state.model

    # Extract trees as a dataframe
    trees_df = model.trees_to_dataframe()

    n_trees = trees_df["Tree"].nunique()
    total_nodes = len(trees_df)
    total_leaves = len(trees_df[trees_df["Feature"] == "Leaf"])
    max_depth = int(trees_df["Depth"].max()) if "Depth" in trees_df.columns else None

    # Feature importance by type
    importance_types = ["gain", "weight", "cover"]
    feature_importance = {}
    feat_map = {
        "f0": "smart_5_raw",
        "f1": "smart_187_raw",
        "f2": "smart_188_raw",
        "f3": "smart_197_raw",
        "f4": "smart_198_raw",
    }
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

    return {
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

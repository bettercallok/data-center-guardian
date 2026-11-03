"""
api/routes_monitoring.py — Monitoring Routes

GET    /api/models/health          — real-time novelty + per-feature KS/PSI drift scores
GET    /api/models/alerts          — LATENT_NOVELTY and FEATURE_DRIFT alerts
DELETE /api/models/alerts/{id}     — manually resolve an alert          (admin)
GET    /api/models/drift/history   — persisted drift snapshots for trend charts
"""
import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from sqlalchemy.orm import Session

from src.api.config import settings
from src.api.security import limiter, require_admin
from src.database.connection import get_db
from src.database.models import DriftSnapshot
from src.database.repository import persist_alert

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/models", tags=["monitoring"])

_UUID_PATTERN = r"^[0-9a-fA-F-]{36}$"


@router.get("/health")
@limiter.limit(settings.rate_limit_read)
def get_health(request: Request):
    """
    Returns the real-time model health summary:
      - Model load status
      - FAISS index status and novelty threshold
      - Drift detector calibration status
      - Per-feature KS and PSI scores from the current sliding window
      - Active alert count by type
    """
    state = request.app.state

    faiss_status = {
        "is_built": state.faiss_indexer.is_built,
        "threshold": state.faiss_indexer.threshold,
    }

    drift_data = None
    if state.drift_detector.is_calibrated:
        report = state.drift_detector.compute_drift()
        if report:
            drift_data = {
                "overall_drift_detected": report.overall_drift_detected,
                "drifting_features": report.drifting_features,
                "window_size": report.window_size,
                "feature_scores": [
                    {
                        "feature": s.feature,
                        "ks_statistic": s.ks_statistic,
                        "ks_p_value": s.ks_p_value,
                        "psi": s.psi,
                        "severity": s.severity,
                        "is_drifting": s.is_drifting,
                    }
                    for s in report.feature_scores
                ],
            }

    active_alerts = state.alert_engine.get_active_alerts()
    alert_summary = {
        "total_active": len(active_alerts),
        "latent_novelty": sum(1 for a in active_alerts if a.alert_type == "LATENT_NOVELTY"),
        "feature_drift": sum(1 for a in active_alerts if a.alert_type == "FEATURE_DRIFT"),
        "critical_count": sum(1 for a in active_alerts if a.severity == "CRITICAL"),
    }

    model_loaded = state.model is not None
    if not model_loaded:
        status = "unavailable"
    elif alert_summary["total_active"] > 0:
        status = "degraded"
    else:
        status = "healthy"

    fp = state.fingerprint
    return {
        "status": status,
        "model_loaded": model_loaded,
        "fingerprint_id": fp.fingerprint_id if fp else None,
        "total_inferences": state.inference_count,
        "drift_window_count": state.drift_detector.window_count,
        "faiss": faiss_status,
        "drift": drift_data,
        "alerts": alert_summary,
    }


@router.get("/alerts")
@limiter.limit(settings.rate_limit_read)
def get_alerts(
    request: Request,
    status: Literal["active", "all"] = "active",
    limit: int = Query(50, ge=1, le=500),
):
    """Returns alerts filtered by status (active|all), newest first."""
    engine = request.app.state.alert_engine
    alerts = engine.get_active_alerts() if status == "active" else engine.get_all_alerts(limit=limit)
    return [a.to_dict() for a in alerts[:limit]]


@router.delete("/alerts/{alert_id}", dependencies=[Depends(require_admin)])
@limiter.limit("30/minute")
def resolve_alert(
    request: Request,
    alert_id: str = Path(..., pattern=_UUID_PATTERN),
    db: Session = Depends(get_db),
):
    """Manually resolve an alert by its ID. Requires the admin API key."""
    alert = request.app.state.alert_engine.resolve_alert(alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found.")
    persist_alert(db, alert)
    return {"status": "resolved", "alert_id": alert_id}


@router.get("/drift/history")
@limiter.limit(settings.rate_limit_read)
def drift_history(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Persisted drift snapshots (one every DRIFT_CHECK_INTERVAL inferences), oldest first."""
    snaps = (
        db.query(DriftSnapshot)
        .order_by(DriftSnapshot.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": s.id,
            "created_at": s.created_at.isoformat() if s.created_at else None,
            "window_size": s.window_size,
            "overall_drift_detected": s.overall_drift_detected,
            "drifting_features": s.drifting_features or [],
            "feature_scores": s.feature_scores_json or [],
        }
        for s in reversed(snaps)
    ]

"""
api/routes_monitoring.py — Monitoring Routes

GET /api/models/health       — real-time novelty rate + per-feature KS/PSI drift scores
GET /api/models/alerts       — active LATENT_NOVELTY and FEATURE_DRIFT alerts
DELETE /api/models/alerts/{id} — manually resolve an alert
"""
import logging
from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/models", tags=["monitoring"])


@router.get("/health")
async def get_health(request: Request):
    """
    Returns the real-time model health summary:
      - FAISS index status and novelty threshold
      - Drift detector calibration status
      - Per-feature KS and PSI scores from the current sliding window
      - Active alert count by type
    """
    state = request.app.state

    # FAISS status
    faiss_status = {
        "is_built": state.faiss_indexer.is_built,
        "threshold": state.faiss_indexer.threshold,
    }

    # Drift report
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

    # Alert summary
    active_alerts = state.alert_engine.get_active_alerts()
    alert_summary = {
        "total_active": len(active_alerts),
        "latent_novelty": sum(1 for a in active_alerts if a.alert_type == "LATENT_NOVELTY"),
        "feature_drift": sum(1 for a in active_alerts if a.alert_type == "FEATURE_DRIFT"),
        "critical_count": sum(1 for a in active_alerts if a.severity == "CRITICAL"),
    }

    return {
        "status": "degraded" if alert_summary["total_active"] > 0 else "healthy",
        "total_inferences": state.inference_count,
        "faiss": faiss_status,
        "drift": drift_data,
        "alerts": alert_summary,
    }


@router.get("/alerts")
async def get_alerts(request: Request, status: str = "active", limit: int = 50):
    """
    Returns alerts filtered by status.
    Query param: status=active|all (default: active)
    """
    state = request.app.state

    if status == "active":
        alerts = state.alert_engine.get_active_alerts()
    else:
        alerts = state.alert_engine.get_all_alerts(limit=limit)

    return [
        {
            "id": a.id,
            "alert_type": a.alert_type,
            "severity": a.severity,
            "status": a.status,
            "message": a.message,
            "metadata": a.metadata,
            "created_at": a.created_at.isoformat(),
            "resolved_at": a.resolved_at.isoformat() if a.resolved_at else None,
            "escalation_count": a.escalation_count,
        }
        for a in alerts[:limit]
    ]


@router.delete("/alerts/{alert_id}")
async def resolve_alert(request: Request, alert_id: str):
    """Manually resolve an alert by its ID."""
    state = request.app.state
    resolved = state.alert_engine.resolve_alert(alert_id)
    if not resolved:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found.")
    return {"status": "resolved", "alert_id": alert_id}

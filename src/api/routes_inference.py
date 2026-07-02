"""
api/routes_inference.py — Enhanced Inference Route

POST /api/models/predict
  - Runs XGBoost AFT survival inference
  - Scores the data point for LATENT_NOVELTY via FAISS
  - Observes the data point into the drift detector window
  - Persists the inference record to the database
  - Fires or resolves alerts accordingly
"""
import logging
import math
import numpy as np

from fastapi import APIRouter, HTTPException, Request, Depends
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

from src.api.schemas import (
    DriveTelemetry,
    EnrichedPrediction,
)
from src.database.connection import get_db
from src.database.models import InferenceRecord

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/models", tags=["inference"])
limiter = Limiter(key_func=get_remote_address)


def _get_risk_level(ttf_days: float) -> str:
    if ttf_days > 1000:
        return "low"
    elif ttf_days > 365:
        return "medium"
    elif ttf_days > 90:
        return "high"
    return "critical"


@router.post("/predict", response_model=EnrichedPrediction)
@limiter.limit("60/minute")
async def predict(request: Request, telemetry: DriveTelemetry, db: Session = Depends(get_db)):
    """
    Run survival inference on a single drive's SMART telemetry.

    The response is enriched with:
      - novelty_score: normalized distance in FAISS latent space (>1.0 = novel)
      - is_novel: True if the point is outside the training distribution
      - drift_report: current KS/PSI drift summary
      - fired_alert_ids: IDs of any alerts triggered by this inference
    """
    state = request.app.state

    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not available.")

    X = np.array([[
        telemetry.smart_5_raw,
        telemetry.smart_187_raw,
        telemetry.smart_188_raw,
        telemetry.smart_197_raw,
        telemetry.smart_198_raw,
    ]], dtype=np.float32)

    import xgboost as xgb
    dmatrix = xgb.DMatrix(X)

    try:
        output = state.model.predict(dmatrix)
        ttf_days = float(output[0])
    except Exception:
        logger.error("Inference failed", exc_info=True)
        raise HTTPException(status_code=500, detail="Inference failed.")

    risk_level = _get_risk_level(ttf_days)
    rul_days = round(max(0, ttf_days - 30), 1)
    log_time = round(math.log(max(1.0, ttf_days)), 4)

    # --- Novelty scoring ---
    novelty_score = None
    is_novel = False
    novelty_distance = None
    fired_alert_ids = []

    if state.faiss_indexer.is_built:
        novelty_result = state.faiss_indexer.score(state.model, X)
        novelty_score = novelty_result.novelty_score
        is_novel = novelty_result.is_novel
        novelty_distance = novelty_result.distance

        if is_novel:
            alert = state.alert_engine.fire_novelty(
                novelty_score=novelty_score,
                distance=novelty_distance,
                threshold=novelty_result.threshold,
            )
            fired_alert_ids.append(alert.id)
        else:
            state.alert_engine.resolve_novelty()

    # --- Drift observation ---
    if state.drift_detector.is_calibrated:
        state.drift_detector.observe(X)

    # --- Drift check (every 50 inferences) ---
    state.inference_count += 1
    drift_summary = None
    if state.inference_count % 50 == 0 and state.drift_detector.is_calibrated:
        drift_report = state.drift_detector.compute_drift()
        if drift_report and drift_report.overall_drift_detected:
            alert = state.alert_engine.fire_drift(
                drifting_features=drift_report.drifting_features,
                drift_details=[
                    {
                        "feature": s.feature,
                        "psi": s.psi,
                        "ks_statistic": s.ks_statistic,
                        "severity": s.severity,
                    }
                    for s in drift_report.feature_scores if s.is_drifting
                ],
            )
            fired_alert_ids.append(alert.id)
        elif drift_report:
            state.alert_engine.resolve_drift()
            drift_summary = {
                "overall_drift_detected": drift_report.overall_drift_detected,
                "drifting_features": drift_report.drifting_features,
                "window_size": drift_report.window_size,
            }

    # --- Persist to database ---
    try:
        record = InferenceRecord(
            smart_5_raw=telemetry.smart_5_raw,
            smart_187_raw=telemetry.smart_187_raw,
            smart_188_raw=telemetry.smart_188_raw,
            smart_197_raw=telemetry.smart_197_raw,
            smart_198_raw=telemetry.smart_198_raw,
            ttf_days=round(ttf_days, 1),
            rul_days=rul_days,
            risk_level=risk_level,
            log_time=log_time,
            novelty_score=novelty_score,
            novelty_distance=novelty_distance,
            is_novel=is_novel,
            alert_ids=",".join(fired_alert_ids) if fired_alert_ids else None,
        )
        db.add(record)
        db.commit()
    except Exception:
        logger.error("Failed to persist inference record", exc_info=True)
        db.rollback()

    return EnrichedPrediction(
        ttf_days=round(ttf_days, 1),
        rul_days=rul_days,
        risk_level=risk_level,
        log_time=log_time,
        novelty_score=novelty_score,
        novelty_distance=novelty_distance,
        is_novel=is_novel,
        fired_alert_ids=fired_alert_ids,
        drift_summary=drift_summary,
    )


@router.get("/predictions")
async def get_inference_timeline(limit: int = 100, db: Session = Depends(get_db)):
    """Return the last N inference records ordered by time desc."""
    records = (
        db.query(InferenceRecord)
        .order_by(InferenceRecord.created_at.desc())
        .limit(min(limit, 500))
        .all()
    )
    return [
        {
            "id": r.id,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "ttf_days": r.ttf_days,
            "rul_days": r.rul_days,
            "risk_level": r.risk_level,
            "novelty_score": r.novelty_score,
            "is_novel": r.is_novel,
            "smart_5_raw": r.smart_5_raw,
            "smart_187_raw": r.smart_187_raw,
            "smart_197_raw": r.smart_197_raw,
        }
        for r in records
    ]

"""
api/inference_service.py — Shared inference + monitoring pipeline

A single implementation used by the single-drive, batch and legacy prediction
endpoints, so all three apply identical novelty, drift, alerting and
persistence semantics.
"""
import logging
import math
import threading
from typing import Sequence

import numpy as np
import xgboost as xgb
from fastapi import HTTPException
from sqlalchemy.orm import Session

from src.api.config import settings
from src.api.schemas import DriveTelemetry, EnrichedPrediction
from src.database.models import InferenceRecord
from src.database.repository import persist_alert, persist_drift_snapshot

logger = logging.getLogger(__name__)

FEATURE_ORDER = ("smart_5_raw", "smart_187_raw", "smart_188_raw", "smart_197_raw", "smart_198_raw")

# Remaining useful life is reported with a 30-day maintenance safety buffer:
# operators should act *before* the predicted failure, not on the day of it.
MAINTENANCE_BUFFER_DAYS = 30

# Guards mutation of shared monitoring state (drift window, counters, alerts)
# when handlers run concurrently in the threadpool.
_state_lock = threading.Lock()


def get_risk_level(ttf_days: float) -> str:
    if ttf_days > 1000:
        return "low"
    elif ttf_days > 365:
        return "medium"
    elif ttf_days > 90:
        return "high"
    return "critical"


def telemetry_matrix(drives: Sequence[DriveTelemetry]) -> np.ndarray:
    return np.array(
        [[getattr(d, f) for f in FEATURE_ORDER] for d in drives],
        dtype=np.float32,
    )


def _drift_dict(report) -> dict:
    return {
        "overall_drift_detected": report.overall_drift_detected,
        "drifting_features": report.drifting_features,
        "window_size": report.window_size,
        "feature_scores": [
            {
                "feature": s.feature,
                "psi": s.psi,
                "ks_statistic": s.ks_statistic,
                "severity": s.severity,
                "is_drifting": s.is_drifting,
            }
            for s in report.feature_scores
        ],
    }


def run_inference(state, drives: Sequence[DriveTelemetry], db: Session) -> list[EnrichedPrediction]:
    """Score drives, update monitoring state, persist records, and fire alerts."""
    if getattr(state, "model", None) is None:
        raise HTTPException(status_code=503, detail="Model not available.")

    X = telemetry_matrix(drives)
    try:
        ttf_preds = np.atleast_1d(state.model.predict(xgb.DMatrix(X)))
    except Exception:
        logger.error("Inference failed", exc_info=True)
        raise HTTPException(status_code=500, detail="Inference failed.")

    novelty_results = None
    if state.faiss_indexer.is_built:
        try:
            novelty_results = state.faiss_indexer.score_batch(state.model, X)
        except Exception:
            logger.error("Novelty scoring failed", exc_info=True)

    changed_alerts = {}
    drift_summary = None
    drift_report_to_persist = None
    per_drive_alerts: list[list[str]] = [[] for _ in drives]

    with _state_lock:
        for i in range(len(drives)):
            if novelty_results is not None:
                nr = novelty_results[i]
                if nr.is_novel:
                    alert = state.alert_engine.fire_novelty(
                        novelty_score=nr.novelty_score,
                        distance=nr.distance,
                        threshold=nr.threshold,
                        out_of_range_features=nr.out_of_range_features,
                    )
                    changed_alerts[alert.id] = alert
                    per_drive_alerts[i].append(alert.id)
                else:
                    state.alert_engine.resolve_novelty()

            if state.drift_detector.is_calibrated:
                state.drift_detector.observe(X[i:i + 1])

            state.inference_count += 1
            interval = max(1, settings.drift_check_interval)
            if state.inference_count % interval == 0 and state.drift_detector.is_calibrated:
                report = state.drift_detector.compute_drift()
                if report is not None:
                    drift_summary = _drift_dict(report)
                    drift_report_to_persist = report
                    if report.overall_drift_detected:
                        alert = state.alert_engine.fire_drift(
                            drifting_features=report.drifting_features,
                            drift_details=[
                                {
                                    "feature": s.feature,
                                    "psi": s.psi,
                                    "ks_statistic": s.ks_statistic,
                                    "severity": s.severity,
                                }
                                for s in report.feature_scores if s.is_drifting
                            ],
                        )
                        changed_alerts[alert.id] = alert
                        per_drive_alerts[i].append(alert.id)
                    else:
                        state.alert_engine.resolve_drift()

    # --- Build responses and persist ---
    predictions: list[EnrichedPrediction] = []
    records: list[InferenceRecord] = []
    for i, tel in enumerate(drives):
        ttf_days = float(ttf_preds[i])
        if not math.isfinite(ttf_days):
            logger.error("Model produced non-finite prediction for input %s", X[i].tolist())
            raise HTTPException(status_code=500, detail="Inference failed.")
        ttf_days = max(0.0, ttf_days)
        rul_days = round(max(0.0, ttf_days - MAINTENANCE_BUFFER_DAYS), 1)
        log_time = round(math.log(max(1.0, ttf_days)), 4)
        risk_level = get_risk_level(ttf_days)
        nr = novelty_results[i] if novelty_results is not None else None

        record = InferenceRecord(
            serial_number=tel.serial_number,
            drive_model=tel.drive_model,
            smart_5_raw=tel.smart_5_raw,
            smart_187_raw=tel.smart_187_raw,
            smart_188_raw=tel.smart_188_raw,
            smart_197_raw=tel.smart_197_raw,
            smart_198_raw=tel.smart_198_raw,
            ttf_days=round(ttf_days, 1),
            rul_days=rul_days,
            risk_level=risk_level,
            log_time=log_time,
            novelty_score=nr.novelty_score if nr else None,
            novelty_distance=nr.distance if nr else None,
            is_novel=nr.is_novel if nr else False,
            alert_ids=",".join(per_drive_alerts[i]) or None,
        )
        records.append(record)

        predictions.append(EnrichedPrediction(
            serial_number=tel.serial_number,
            ttf_days=round(ttf_days, 1),
            rul_days=rul_days,
            risk_level=risk_level,
            log_time=log_time,
            novelty_score=nr.novelty_score if nr else None,
            novelty_distance=nr.distance if nr else None,
            is_novel=nr.is_novel if nr else False,
            out_of_range_features=nr.out_of_range_features if nr else [],
            fired_alert_ids=per_drive_alerts[i],
            drift_summary=drift_summary if i == len(drives) - 1 else None,
        ))

    try:
        db.add_all(records)
        db.commit()
        for rec, pred in zip(records, predictions):
            pred.id = rec.id
    except Exception:
        logger.error("Failed to persist inference records", exc_info=True)
        db.rollback()

    for alert in changed_alerts.values():
        persist_alert(db, alert)
    if drift_report_to_persist is not None:
        persist_drift_snapshot(db, drift_report_to_persist)

    return predictions

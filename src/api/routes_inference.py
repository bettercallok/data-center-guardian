"""
api/routes_inference.py — Inference Routes

POST /api/models/predict              — score one drive (novelty + drift + alerts)
POST /api/models/predict/batch        — score up to MAX_BATCH_SIZE drives at once
GET  /api/models/predictions          — recent inference timeline (filterable)
GET  /api/models/predictions/export   — the same timeline as a CSV download
GET  /api/models/predictions/summary  — risk-tier / novelty aggregate stats
GET  /api/drives/{serial}/history     — prediction history of a single drive

Handlers are plain `def` functions: FastAPI runs them in a worker threadpool so
CPU-bound XGBoost/FAISS work never blocks the event loop.
"""
import csv
import io
import logging
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.api.config import settings
from src.api.inference_service import run_inference
from src.api.schemas import (
    BatchPredictionResponse,
    BatchTelemetry,
    DriveTelemetry,
    EnrichedPrediction,
    _IDENT_PATTERN,
)
from src.api.security import limiter
from src.database.connection import get_db
from src.database.models import InferenceRecord

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/models", tags=["inference"])
drives_router = APIRouter(prefix="/api/drives", tags=["drives"])

RiskFilter = Optional[Literal["low", "medium", "high", "critical"]]

CSV_COLUMNS = [
    "id", "created_at", "serial_number", "drive_model",
    "smart_5_raw", "smart_187_raw", "smart_188_raw", "smart_197_raw", "smart_198_raw",
    "ttf_days", "rul_days", "risk_level", "novelty_score", "is_novel",
]


def _record_to_dict(r: InferenceRecord) -> dict:
    return {
        "id": r.id,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "serial_number": r.serial_number,
        "drive_model": r.drive_model,
        "ttf_days": r.ttf_days,
        "rul_days": r.rul_days,
        "risk_level": r.risk_level,
        "novelty_score": r.novelty_score,
        "is_novel": r.is_novel,
        "smart_5_raw": r.smart_5_raw,
        "smart_187_raw": r.smart_187_raw,
        "smart_188_raw": r.smart_188_raw,
        "smart_197_raw": r.smart_197_raw,
        "smart_198_raw": r.smart_198_raw,
    }


def _csv_safe(value) -> str:
    """Neutralise spreadsheet formula injection (CWE-1236)."""
    if value is None:
        return ""
    text = str(value)
    if text and text[0] in ("=", "+", "-", "@", "\t", "\r"):
        # Negative numbers are legitimate; only prefix non-numeric text.
        try:
            float(text)
        except ValueError:
            return "'" + text
    return text


def _filtered_query(db: Session, risk_level: RiskFilter, serial_number: Optional[str]):
    q = db.query(InferenceRecord)
    if risk_level:
        q = q.filter(InferenceRecord.risk_level == risk_level)
    if serial_number:
        q = q.filter(InferenceRecord.serial_number == serial_number)
    return q.order_by(InferenceRecord.created_at.desc())


@router.post("/predict", response_model=EnrichedPrediction)
@limiter.limit(settings.rate_limit_predict)
def predict(request: Request, telemetry: DriveTelemetry, db: Session = Depends(get_db)):
    """
    Run survival inference on a single drive's SMART telemetry.

    The response is enriched with:
      - novelty_score: normalized novelty (>1.0 = novel)
      - is_novel / out_of_range_features: why the point was flagged
      - drift_summary: KS/PSI drift summary when computed on this call
      - fired_alert_ids: IDs of any alerts triggered by this inference
    """
    return run_inference(request.app.state, [telemetry], db)[0]


@router.post("/predict/batch", response_model=BatchPredictionResponse)
@limiter.limit(settings.rate_limit_batch)
def predict_batch(request: Request, batch: BatchTelemetry, db: Session = Depends(get_db)):
    """Score a batch of drives in a single model call (fleet scans, CSV uploads)."""
    if len(batch.drives) > settings.max_batch_size:
        raise HTTPException(
            status_code=413,
            detail=f"Batch too large: {len(batch.drives)} drives (max {settings.max_batch_size}).",
        )
    preds = run_inference(request.app.state, batch.drives, db)
    risk_counts = {tier: 0 for tier in ("low", "medium", "high", "critical")}
    for p in preds:
        risk_counts[p.risk_level] += 1
    return BatchPredictionResponse(
        count=len(preds),
        risk_counts=risk_counts,
        novel_count=sum(1 for p in preds if p.is_novel),
        predictions=preds,
    )


@router.get("/predictions")
@limiter.limit(settings.rate_limit_read)
def get_inference_timeline(
    request: Request,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0, le=100_000),
    risk_level: RiskFilter = None,
    serial_number: Optional[str] = Query(None, pattern=_IDENT_PATTERN),
    db: Session = Depends(get_db),
):
    """Return the most recent inference records, newest first."""
    records = _filtered_query(db, risk_level, serial_number).offset(offset).limit(limit).all()
    return [_record_to_dict(r) for r in records]


@router.get("/predictions/export")
@limiter.limit("10/minute")
def export_predictions_csv(
    request: Request,
    limit: int = Query(1000, ge=1, le=10_000),
    risk_level: RiskFilter = None,
    serial_number: Optional[str] = Query(None, pattern=_IDENT_PATTERN),
    db: Session = Depends(get_db),
):
    """Download the inference timeline as CSV."""
    records = _filtered_query(db, risk_level, serial_number).limit(limit).all()
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(CSV_COLUMNS)
    for r in records:
        row = _record_to_dict(r)
        writer.writerow([_csv_safe(row[c]) for c in CSV_COLUMNS])
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="guardian_predictions.csv"'},
    )


@router.get("/predictions/summary")
@limiter.limit(settings.rate_limit_read)
def predictions_summary(request: Request, db: Session = Depends(get_db)):
    """Aggregate statistics over all persisted predictions."""
    total = db.query(func.count(InferenceRecord.id)).scalar() or 0
    by_risk = dict(
        db.query(InferenceRecord.risk_level, func.count(InferenceRecord.id))
        .group_by(InferenceRecord.risk_level)
        .all()
    )
    novel = db.query(func.count(InferenceRecord.id)).filter(InferenceRecord.is_novel.is_(True)).scalar() or 0
    avg_rul = db.query(func.avg(InferenceRecord.rul_days)).scalar()
    distinct_drives = (
        db.query(func.count(func.distinct(InferenceRecord.serial_number)))
        .filter(InferenceRecord.serial_number.isnot(None))
        .scalar() or 0
    )
    return {
        "total_predictions": total,
        "risk_counts": {tier: int(by_risk.get(tier, 0)) for tier in ("low", "medium", "high", "critical")},
        "novel_count": novel,
        "novelty_rate": round(novel / total, 4) if total else 0.0,
        "mean_rul_days": round(float(avg_rul), 1) if avg_rul is not None else None,
        "tracked_drives": distinct_drives,
    }


@drives_router.get("/{serial_number}/history")
@limiter.limit(settings.rate_limit_read)
def drive_history(
    request: Request,
    serial_number: str = Path(..., pattern=_IDENT_PATTERN),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Prediction history for one drive, newest first, with an RUL trend."""
    records = _filtered_query(db, None, serial_number).limit(limit).all()
    if not records:
        raise HTTPException(status_code=404, detail="No predictions recorded for this drive.")
    history = [_record_to_dict(r) for r in records]
    trend = None
    if len(records) >= 2:
        # Positive = RUL improving, negative = degrading (newest minus oldest in window)
        trend = round(records[0].rul_days - records[-1].rul_days, 1)
    return {
        "serial_number": serial_number,
        "drive_model": next((r.drive_model for r in records if r.drive_model), None),
        "n_predictions": len(history),
        "latest": history[0],
        "rul_trend_days": trend,
        "history": history,
    }

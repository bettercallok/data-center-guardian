"""
database/models.py — SQLAlchemy ORM Models

Persistent storage for inference history, alerts, and drift snapshots.
Uses SQLite by default (via DATABASE_URL env var), with full Postgres compat.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, JSON, Text
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


def _utcnow():
    return datetime.now(timezone.utc)


class InferenceRecord(Base):
    """
    Persists every inference call with inputs, outputs, and monitoring signals.
    Enables querying the inference timeline for the dashboard.
    """
    __tablename__ = "inference_records"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime(timezone=True), default=_utcnow, index=True)

    # Inputs
    smart_5_raw = Column(Float, nullable=False)
    smart_187_raw = Column(Float, nullable=False)
    smart_188_raw = Column(Float, nullable=False)
    smart_197_raw = Column(Float, nullable=False)
    smart_198_raw = Column(Float, nullable=False)

    # Outputs
    ttf_days = Column(Float, nullable=False)
    rul_days = Column(Float, nullable=False)
    risk_level = Column(String(10), nullable=False)
    log_time = Column(Float, nullable=False)

    # Monitoring signals
    novelty_score = Column(Float, nullable=True)
    novelty_distance = Column(Float, nullable=True)
    is_novel = Column(Boolean, nullable=True)

    # Alert IDs (comma-separated, denormalized for simplicity)
    alert_ids = Column(Text, nullable=True)


class AlertRecord(Base):
    """
    Persists alert lifecycle events for historical review and dashboarding.
    """
    __tablename__ = "alert_records"

    id = Column(String, primary_key=True)
    alert_type = Column(String(30), nullable=False, index=True)
    severity = Column(String(10), nullable=False)
    status = Column(String(10), nullable=False, index=True)
    message = Column(Text, nullable=False)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, index=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    escalation_count = Column(Integer, default=0)


class DriftSnapshot(Base):
    """
    A point-in-time snapshot of the drift report. Stored periodically
    (e.g., every 50 inferences) to enable trend analysis.
    """
    __tablename__ = "drift_snapshots"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime(timezone=True), default=_utcnow, index=True)
    window_size = Column(Integer, nullable=False)
    overall_drift_detected = Column(Boolean, nullable=False)
    drifting_features = Column(JSON, nullable=True)
    feature_scores_json = Column(JSON, nullable=True)

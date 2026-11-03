"""
database/repository.py — Persistence helpers

Keeps SQLAlchemy details out of the route handlers. Every helper swallows and
logs database errors: monitoring persistence must never break inference.
"""
import logging
from typing import Optional

from sqlalchemy.orm import Session

from src.database.models import AlertRecord, DriftSnapshot
from src.monitoring.alert_engine import Alert, AlertSeverity, AlertStatus, AlertType

logger = logging.getLogger(__name__)


def persist_alert(db: Session, alert: Alert) -> None:
    """Insert or update the database row mirroring an in-memory alert."""
    try:
        db.merge(AlertRecord(
            id=alert.id,
            alert_type=alert.alert_type.value,
            severity=alert.severity.value,
            status=alert.status.value,
            message=alert.message,
            metadata_json=alert.metadata,
            created_at=alert.created_at,
            updated_at=alert.updated_at,
            resolved_at=alert.resolved_at,
            escalation_count=alert.escalation_count,
            occurrences=alert.occurrences,
        ))
        db.commit()
    except Exception:
        logger.error("Failed to persist alert %s", alert.id, exc_info=True)
        db.rollback()


def load_active_alerts(db: Session, limit: int = 500) -> list[Alert]:
    """Load ACTIVE alerts so they survive a server restart."""
    try:
        rows = (
            db.query(AlertRecord)
            .filter(AlertRecord.status == AlertStatus.ACTIVE.value)
            .order_by(AlertRecord.created_at.desc())
            .limit(limit)
            .all()
        )
    except Exception:
        logger.error("Failed to load persisted alerts", exc_info=True)
        return []

    alerts = []
    for r in rows:
        try:
            alerts.append(Alert(
                id=r.id,
                alert_type=AlertType(r.alert_type),
                severity=AlertSeverity(r.severity),
                status=AlertStatus(r.status),
                message=r.message,
                metadata=r.metadata_json or {},
                created_at=r.created_at,
                updated_at=r.updated_at,
                resolved_at=r.resolved_at,
                escalation_count=r.escalation_count or 0,
                occurrences=r.occurrences or 1,
            ))
        except ValueError:
            logger.warning("Skipping persisted alert %s with unknown enum value", r.id)
    return alerts


def persist_drift_snapshot(db: Session, report) -> Optional[str]:
    """Store a point-in-time drift report for trend analysis."""
    try:
        snap = DriftSnapshot(
            window_size=report.window_size,
            overall_drift_detected=report.overall_drift_detected,
            drifting_features=report.drifting_features,
            feature_scores_json=[
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
        )
        db.add(snap)
        db.commit()
        return snap.id
    except Exception:
        logger.error("Failed to persist drift snapshot", exc_info=True)
        db.rollback()
        return None

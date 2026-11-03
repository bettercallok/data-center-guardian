"""
alert_engine.py — Alert Generation and Severity Escalation

Manages the lifecycle of monitoring alerts:
  - LATENT_NOVELTY: fired when a data point's FAISS distance exceeds the threshold.
  - FEATURE_DRIFT: fired when KS/PSI drift is detected in one or more features.

Alerts are stored in-memory and persisted to the database by the API layer.
Each alert has a severity (WARNING, CRITICAL) that escalates if the condition
persists across multiple consecutive inferences.

De-duplication: while an alert of a given type is ACTIVE, repeated firings
update and escalate that alert instead of creating a new one. This keeps the
store bounded no matter how many anomalous requests arrive.
"""
import logging
import threading
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class AlertType(str, Enum):
    LATENT_NOVELTY = "LATENT_NOVELTY"
    FEATURE_DRIFT = "FEATURE_DRIFT"


class AlertSeverity(str, Enum):
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertStatus(str, Enum):
    ACTIVE = "ACTIVE"
    RESOLVED = "RESOLVED"


@dataclass
class Alert:
    """Represents a single monitoring alert."""
    id: str
    alert_type: AlertType
    severity: AlertSeverity
    status: AlertStatus
    message: str
    metadata: dict
    created_at: datetime
    resolved_at: Optional[datetime] = None
    escalation_count: int = 0  # how many repeat firings kept this alive
    updated_at: Optional[datetime] = None
    occurrences: int = 1

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "status": self.status.value,
            "message": self.message,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "escalation_count": self.escalation_count,
            "occurrences": self.occurrences,
        }


class AlertEngine:
    """
    Thread-safe in-memory alert store with de-duplication and severity escalation.

    Escalation rule: if the same alert type fires 3 consecutive times
    without resolving, it escalates from WARNING to CRITICAL. An active alert
    is never de-escalated; it must be resolved explicitly.
    """

    ESCALATION_THRESHOLD = 3

    def __init__(self, max_alerts: int = 500):
        self._alerts: dict[str, Alert] = {}
        self._max_alerts = max_alerts
        self._consecutive: dict[AlertType, int] = {t: 0 for t in AlertType}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Firing
    # ------------------------------------------------------------------

    def fire_novelty(self, novelty_score: float, distance: float, threshold: float,
                     out_of_range_features: Optional[list[str]] = None) -> Alert:
        """Create or escalate a LATENT_NOVELTY alert."""
        oor = out_of_range_features or []
        message = (
            f"Model received data outside its training distribution. "
            f"Novelty score: {novelty_score:.2f} (threshold: 1.00). "
            f"Leaf-space distance: {distance:.4f} vs threshold {threshold:.4f}."
        )
        if oor:
            message += f" Out-of-range features: [{', '.join(oor)}]."
        return self._fire(
            AlertType.LATENT_NOVELTY,
            message,
            {
                "novelty_score": novelty_score,
                "distance": distance,
                "threshold": threshold,
                "out_of_range_features": oor,
            },
        )

    def fire_drift(self, drifting_features: list[str], drift_details: list[dict]) -> Alert:
        """Create or escalate a FEATURE_DRIFT alert."""
        feat_str = ", ".join(drifting_features)
        return self._fire(
            AlertType.FEATURE_DRIFT,
            (
                f"Statistical drift detected in features: [{feat_str}]. "
                f"KS/PSI thresholds exceeded. Model predictions may be unreliable."
            ),
            {"drifting_features": drifting_features, "drift_details": drift_details},
        )

    def _fire(self, alert_type: AlertType, message: str, metadata: dict) -> Alert:
        with self._lock:
            self._consecutive[alert_type] += 1
            consecutive = self._consecutive[alert_type]
            severity = (
                AlertSeverity.CRITICAL
                if consecutive >= self.ESCALATION_THRESHOLD
                else AlertSeverity.WARNING
            )
            metadata = {**metadata, "consecutive_count": consecutive}
            now = datetime.now(timezone.utc)

            existing = self._active_of_type(alert_type)
            if existing is not None:
                existing.occurrences += 1
                existing.escalation_count += 1
                existing.message = message
                existing.metadata = metadata
                existing.updated_at = now
                if severity == AlertSeverity.CRITICAL:
                    existing.severity = AlertSeverity.CRITICAL
                alert = existing
            else:
                alert = Alert(
                    id=str(uuid.uuid4()),
                    alert_type=alert_type,
                    severity=severity,
                    status=AlertStatus.ACTIVE,
                    message=message,
                    metadata=metadata,
                    created_at=now,
                    updated_at=now,
                    escalation_count=max(0, consecutive - 1),
                )
                self._store(alert)

        logger.warning(
            "ALERT %s [%s][%s] consecutive=%d occurrences=%d",
            "FIRED" if alert.occurrences == 1 else "UPDATED",
            alert.alert_type.value, alert.severity.value, consecutive, alert.occurrences,
        )
        return alert

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve_novelty(self) -> None:
        """Reset novelty streak when a non-novel inference occurs."""
        with self._lock:
            self._consecutive[AlertType.LATENT_NOVELTY] = 0

    def resolve_drift(self) -> None:
        """Reset drift streak when drift is not detected."""
        with self._lock:
            self._consecutive[AlertType.FEATURE_DRIFT] = 0

    def resolve_alert(self, alert_id: str) -> Optional[Alert]:
        """Manually resolve an alert by ID. Returns the alert if found."""
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert is None:
                return None
            if alert.status != AlertStatus.RESOLVED:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now(timezone.utc)
                alert.updated_at = alert.resolved_at
                self._consecutive[alert.alert_type] = 0
            logger.info("Alert %s resolved.", alert_id)
            return alert

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_active_alerts(self) -> list[Alert]:
        with self._lock:
            return sorted(
                (a for a in self._alerts.values() if a.status == AlertStatus.ACTIVE),
                key=lambda a: a.created_at,
                reverse=True,
            )

    def get_all_alerts(self, limit: int = 100) -> list[Alert]:
        with self._lock:
            all_alerts = sorted(self._alerts.values(), key=lambda a: a.created_at, reverse=True)
        return all_alerts[:limit]

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        with self._lock:
            return self._alerts.get(alert_id)

    def restore(self, alert: Alert) -> None:
        """Re-hydrate a persisted alert (used on startup)."""
        with self._lock:
            self._store(alert)

    def __len__(self) -> int:
        with self._lock:
            return len(self._alerts)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _active_of_type(self, alert_type: AlertType) -> Optional[Alert]:
        for a in self._alerts.values():
            if a.alert_type == alert_type and a.status == AlertStatus.ACTIVE:
                return a
        return None

    def _store(self, alert: Alert) -> None:
        """Insert an alert, evicting the oldest (resolved first) to respect the cap."""
        while len(self._alerts) >= self._max_alerts:
            resolved = [a for a in self._alerts.values() if a.status == AlertStatus.RESOLVED]
            pool = resolved or list(self._alerts.values())
            oldest = min(pool, key=lambda a: a.created_at)
            del self._alerts[oldest.id]
        self._alerts[alert.id] = alert

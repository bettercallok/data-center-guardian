"""
alert_engine.py — Alert Generation and Severity Escalation

Manages the lifecycle of monitoring alerts:
  - LATENT_NOVELTY: fired when a data point's FAISS distance exceeds the threshold.
  - FEATURE_DRIFT: fired when KS/PSI drift is detected in one or more features.

Alerts are stored in-memory (and persisted to the database via the db session).
Each alert has a severity (WARNING, CRITICAL) that escalates if the condition
persists across multiple consecutive inferences.
"""
import logging
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
    escalation_count: int = 0  # how many consecutive inferences kept this alive


class AlertEngine:
    """
    In-memory alert store with severity escalation logic.

    Escalation rule: if the same alert type fires 3 consecutive times
    without resolving, it escalates from WARNING to CRITICAL.
    """

    def __init__(self, max_active_alerts: int = 500):
        self._alerts: dict[str, Alert] = {}
        self._max_active = max_active_alerts
        self._consecutive_novelty: int = 0
        self._consecutive_drift: int = 0

    # ------------------------------------------------------------------
    # Firing
    # ------------------------------------------------------------------

    def fire_novelty(self, novelty_score: float, distance: float, threshold: float) -> Alert:
        """Create or escalate a LATENT_NOVELTY alert."""
        self._consecutive_novelty += 1

        severity = (
            AlertSeverity.CRITICAL
            if self._consecutive_novelty >= 3
            else AlertSeverity.WARNING
        )

        alert = Alert(
            id=str(uuid.uuid4()),
            alert_type=AlertType.LATENT_NOVELTY,
            severity=severity,
            status=AlertStatus.ACTIVE,
            message=(
                f"Model received data outside its training distribution. "
                f"Novelty score: {novelty_score:.2f} (threshold: 1.00). "
                f"L2 distance: {distance:.4f} vs threshold {threshold:.4f}."
            ),
            metadata={
                "novelty_score": novelty_score,
                "distance": distance,
                "threshold": threshold,
                "consecutive_count": self._consecutive_novelty,
            },
            created_at=datetime.now(timezone.utc),
            escalation_count=max(0, self._consecutive_novelty - 1),
        )
        self._store(alert)
        logger.warning(
            "ALERT FIRED [%s][%s] novelty_score=%.2f consecutive=%d",
            alert.alert_type, alert.severity, novelty_score, self._consecutive_novelty
        )
        return alert

    def fire_drift(self, drifting_features: list[str], drift_details: list[dict]) -> Alert:
        """Create or escalate a FEATURE_DRIFT alert."""
        self._consecutive_drift += 1

        severity = (
            AlertSeverity.CRITICAL
            if self._consecutive_drift >= 3
            else AlertSeverity.WARNING
        )

        feat_str = ", ".join(drifting_features)
        alert = Alert(
            id=str(uuid.uuid4()),
            alert_type=AlertType.FEATURE_DRIFT,
            severity=severity,
            status=AlertStatus.ACTIVE,
            message=(
                f"Statistical drift detected in features: [{feat_str}]. "
                f"KS/PSI thresholds exceeded. Model predictions may be unreliable."
            ),
            metadata={
                "drifting_features": drifting_features,
                "drift_details": drift_details,
                "consecutive_count": self._consecutive_drift,
            },
            created_at=datetime.now(timezone.utc),
            escalation_count=max(0, self._consecutive_drift - 1),
        )
        self._store(alert)
        logger.warning(
            "ALERT FIRED [%s][%s] features=%s consecutive=%d",
            alert.alert_type, alert.severity, feat_str, self._consecutive_drift
        )
        return alert

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve_novelty(self) -> None:
        """Reset novelty streak when a non-novel inference occurs."""
        self._consecutive_novelty = 0

    def resolve_drift(self) -> None:
        """Reset drift streak when drift is not detected."""
        self._consecutive_drift = 0

    def resolve_alert(self, alert_id: str) -> bool:
        """Manually resolve an alert by ID. Returns True if found."""
        if alert_id in self._alerts:
            self._alerts[alert_id].status = AlertStatus.RESOLVED
            self._alerts[alert_id].resolved_at = datetime.now(timezone.utc)
            logger.info("Alert %s resolved.", alert_id)
            return True
        return False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_active_alerts(self) -> list[Alert]:
        return [a for a in self._alerts.values() if a.status == AlertStatus.ACTIVE]

    def get_all_alerts(self, limit: int = 100) -> list[Alert]:
        all_alerts = sorted(
            self._alerts.values(),
            key=lambda a: a.created_at,
            reverse=True
        )
        return all_alerts[:limit]

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        return self._alerts.get(alert_id)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _store(self, alert: Alert) -> None:
        if len(self._alerts) >= self._max_active:
            # Evict the oldest resolved alert to stay within memory budget
            resolved = [
                a for a in self._alerts.values()
                if a.status == AlertStatus.RESOLVED
            ]
            if resolved:
                oldest = min(resolved, key=lambda a: a.created_at)
                del self._alerts[oldest.id]
        self._alerts[alert.id] = alert

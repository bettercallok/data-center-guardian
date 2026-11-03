"""
api/schemas.py — Pydantic Request/Response Schemas

All public API contract types. Each schema is versioned and documented
to serve as the source of truth for the OpenAPI /docs output.
"""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

RiskLevel = Literal["low", "medium", "high", "critical"]

# Conservative identifier pattern: letters, digits, and a few separators.
# Prevents log/CSV injection and keeps stored identifiers predictable.
_IDENT_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9 ._\-/]{0,63}$"


# ---------------------------------------------------------------------------
# Input Schemas
# ---------------------------------------------------------------------------

class DriveTelemetry(BaseModel):
    """
    SMART telemetry input for a single HDD.
    All values are 16-bit unsigned integers (0–65535).
    Pydantic rejects any request outside these bounds with a 422 error.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "serial_number": "ZA1234XY",
                "drive_model": "ST4000DM000",
                "smart_5_raw": 12,
                "smart_187_raw": 3,
                "smart_188_raw": 0,
                "smart_197_raw": 5,
                "smart_198_raw": 5,
            }
        },
    )

    serial_number: Optional[str] = Field(
        default=None, pattern=_IDENT_PATTERN,
        description="Optional drive serial number, used to track a drive over time."
    )
    drive_model: Optional[str] = Field(
        default=None, pattern=_IDENT_PATTERN,
        description="Optional drive model identifier (e.g. ST4000DM000)."
    )
    smart_5_raw: int = Field(
        default=0, ge=0, le=65535,
        description="Reallocated Sectors Count — indicates platter degradation."
    )
    smart_187_raw: int = Field(
        default=0, ge=0, le=65535,
        description="Reported Uncorrectable Errors — hardware ECC failures."
    )
    smart_188_raw: int = Field(
        default=0, ge=0, le=65535,
        description="Command Timeout — operations aborted due to timeout."
    )
    smart_197_raw: int = Field(
        default=0, ge=0, le=65535,
        description="Current Pending Sector Count — sectors awaiting reallocation."
    )
    smart_198_raw: int = Field(
        default=0, ge=0, le=65535,
        description="Uncorrectable Sector Count — sectors that failed reallocation."
    )


class BatchTelemetry(BaseModel):
    """A batch of drives scored in one request (fleet-scan use case)."""
    drives: list[DriveTelemetry] = Field(
        ..., min_length=1, max_length=1000,
        description="Drives to score (the server enforces MAX_BATCH_SIZE, hard ceiling 1000).",
    )


# ---------------------------------------------------------------------------
# Output Schemas
# ---------------------------------------------------------------------------

class SurvivalPrediction(BaseModel):
    """Legacy prediction response (backward compat with old frontend)."""
    ttf_days: float = Field(..., description="Predicted Time-To-Failure in days.")
    rul_days: float = Field(..., description="Remaining Useful Life in days.")
    risk_level: RiskLevel = Field(..., description="Risk tier: low, medium, high, critical.")
    log_time: float = Field(..., description="Natural log of the predicted survival time.")


class EnrichedPrediction(SurvivalPrediction):
    """
    Full prediction response with monitoring signals.
    Extends SurvivalPrediction with novelty and drift data.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "6f1c0e8e-1d1b-4e6c-9a43-2f1f4a1b9c0d",
                "serial_number": "ZA1234XY",
                "ttf_days": 420.5,
                "rul_days": 390.5,
                "risk_level": "medium",
                "log_time": 6.041,
                "novelty_score": 0.85,
                "novelty_distance": 0.34,
                "is_novel": False,
                "out_of_range_features": [],
                "fired_alert_ids": [],
                "drift_summary": None,
            }
        }
    )

    id: Optional[str] = Field(None, description="ID of the persisted inference record.")
    serial_number: Optional[str] = None

    # Novelty signal
    novelty_score: Optional[float] = Field(
        None,
        description=(
            "Normalized novelty score. >1.0 means the point is outside the "
            "training distribution (leaf-space distance or feature range)."
        )
    )
    novelty_distance: Optional[float] = Field(
        None,
        description="Mean fraction of trees in which the input lands in a different leaf than its nearest probe neighbours (0–1)."
    )
    is_novel: Optional[bool] = Field(
        None,
        description="True if the input is flagged as LATENT_NOVELTY."
    )
    out_of_range_features: list[str] = Field(
        default_factory=list,
        description="Features whose value lies far beyond the probed operating envelope."
    )

    # Alert context
    fired_alert_ids: list[str] = Field(
        default_factory=list,
        description="IDs of any alerts triggered by this inference."
    )

    # Drift context (populated every N inferences)
    drift_summary: Optional[dict] = Field(
        None,
        description="Latest drift report if computed on this inference cycle."
    )


class BatchPredictionResponse(BaseModel):
    count: int
    risk_counts: dict[str, int]
    novel_count: int
    predictions: list[EnrichedPrediction]

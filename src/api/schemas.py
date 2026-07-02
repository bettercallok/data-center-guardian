"""
api/schemas.py — Pydantic Request/Response Schemas

All public API contract types. Each schema is versioned and documented
to serve as the source of truth for the OpenAPI /docs output.
"""
from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Input Schemas
# ---------------------------------------------------------------------------

class DriveTelemetry(BaseModel):
    """
    SMART telemetry input for a single HDD.
    All values are 16-bit unsigned integers (0–65535).
    Pydantic rejects any request outside these bounds with a 422 error.
    """
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

    class Config:
        json_schema_extra = {
            "example": {
                "smart_5_raw": 12,
                "smart_187_raw": 3,
                "smart_188_raw": 0,
                "smart_197_raw": 5,
                "smart_198_raw": 5,
            }
        }


# ---------------------------------------------------------------------------
# Output Schemas
# ---------------------------------------------------------------------------

class SurvivalPrediction(BaseModel):
    """Legacy prediction response (backward compat with old frontend)."""
    ttf_days: float = Field(..., description="Predicted Time-To-Failure in days.")
    rul_days: float = Field(..., description="Remaining Useful Life in days.")
    risk_level: str = Field(..., description="Risk tier: low, medium, high, critical.")
    log_time: float = Field(..., description="Raw log survival time from the AFT model.")


class EnrichedPrediction(BaseModel):
    """
    Full prediction response with monitoring signals.
    Extends SurvivalPrediction with novelty and drift data.
    """
    # Core prediction
    ttf_days: float
    rul_days: float
    risk_level: str
    log_time: float

    # Novelty signal
    novelty_score: Optional[float] = Field(
        None,
        description=(
            "Normalized distance in FAISS latent space. "
            ">1.0 means the point is outside the training distribution."
        )
    )
    novelty_distance: Optional[float] = Field(
        None,
        description="Raw L2 distance in XGBoost leaf-path space."
    )
    is_novel: Optional[bool] = Field(
        None,
        description="True if the input is flagged as LATENT_NOVELTY."
    )

    # Alert context
    fired_alert_ids: list[str] = Field(
        default_factory=list,
        description="IDs of any alerts triggered by this inference."
    )

    # Drift context (populated every 50 inferences)
    drift_summary: Optional[dict] = Field(
        None,
        description="Latest drift report if computed on this inference cycle."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "ttf_days": 420.5,
                "rul_days": 390.5,
                "risk_level": "medium",
                "log_time": 6.041,
                "novelty_score": 0.85,
                "novelty_distance": 0.34,
                "is_novel": False,
                "fired_alert_ids": [],
                "drift_summary": None,
            }
        }

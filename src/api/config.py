"""
api/config.py — Centralised runtime configuration

Every tunable is read from the environment exactly once at import time so
behaviour is explicit and testable. Nothing here should ever hold a default
secret: if a secret is not configured, the feature that needs it fails closed.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_CORS_ORIGINS = (
    "https://data-center-gaurdian.vercel.app",
    "https://bettercallok-data-center-guardian.hf.space",
    "http://localhost:5173",
    "http://localhost:3000",
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: tuple[str, ...]) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return list(default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    # Path to the XGBoost survival model artifact
    model_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("MODEL_PATH", str(PROJECT_ROOT / "src" / "api" / "survival_model.json"))
        )
    )
    telemetry_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("TELEMETRY_PATH", str(PROJECT_ROOT / "data" / "processed" / "telemetry.json"))
        )
    )
    frontend_dist: Path = field(
        default_factory=lambda: Path(os.getenv("FRONTEND_DIST", str(PROJECT_ROOT / "frontend" / "dist")))
    )

    # Admin key protecting state-mutating endpoints (probe sweep, alert resolution).
    # When unset, those endpoints are disabled (fail closed).
    admin_api_key: str = field(default_factory=lambda: os.getenv("GUARDIAN_ADMIN_API_KEY", ""))

    cors_origins: list[str] = field(default_factory=lambda: _env_list("CORS_ORIGINS", DEFAULT_CORS_ORIGINS))

    # Rate limits (slowapi syntax)
    rate_limit_predict: str = field(default_factory=lambda: os.getenv("RATE_LIMIT_PREDICT", "60/minute"))
    rate_limit_batch: str = field(default_factory=lambda: os.getenv("RATE_LIMIT_BATCH", "10/minute"))
    rate_limit_probe: str = field(default_factory=lambda: os.getenv("RATE_LIMIT_PROBE", "5/minute"))
    rate_limit_read: str = field(default_factory=lambda: os.getenv("RATE_LIMIT_READ", "120/minute"))
    rate_limit_enabled: bool = field(default_factory=lambda: _env_bool("RATE_LIMIT_ENABLED", True))

    # Monitoring
    probe_samples_on_startup: int = field(default_factory=lambda: int(os.getenv("PROBE_SAMPLES", "500")))
    drift_check_interval: int = field(default_factory=lambda: int(os.getenv("DRIFT_CHECK_INTERVAL", "50")))
    max_batch_size: int = field(default_factory=lambda: int(os.getenv("MAX_BATCH_SIZE", "100")))

    # Public API docs (/docs, /redoc, /openapi.json). Disable in hardened deployments.
    enable_docs: bool = field(default_factory=lambda: _env_bool("ENABLE_DOCS", True))


settings = Settings()

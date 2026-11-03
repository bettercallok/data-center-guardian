"""
api/main.py — Data Center Guardian API

Enterprise-grade predictive maintenance platform.
On startup, the lifespan handler:
  1. Initializes the database (and restores still-active alerts)
  2. Initializes the FAISS novelty indexer, drift detector, and alert engine
  3. Loads the XGBoost survival model
  4. Automatically runs an LHS probe sweep to build the monitoring baseline

All monitoring is then active from the first inference call.
"""
import hashlib
import json
import logging
from contextlib import asynccontextmanager

import xgboost as xgb
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy.orm import Session

from src.api.config import settings
from src.api.inference_service import run_inference
from src.api.routes_autopsy import router as autopsy_router, run_probe_sweep
from src.api.routes_inference import drives_router, router as inference_router
from src.api.routes_monitoring import router as monitoring_router
from src.api.schemas import DriveTelemetry, SurvivalPrediction
from src.api.security import SecurityHeadersMiddleware, limiter
from src.database.connection import SessionLocal, get_db, init_db
from src.database.repository import load_active_alerts
from src.monitoring.alert_engine import AlertEngine
from src.monitoring.drift_metrics import DriftDetector
from src.monitoring.faiss_indexer import FAISSNoveltyIndexer

API_VERSION = "2.1.0"

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)


def _file_sha256(path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Startup: init db → init monitoring → load model → run probe sweep.
    Shutdown: clean release of all resources.
    """
    logger.info("=" * 60)
    logger.info("DATA CENTER GUARDIAN — STARTUP (v%s)", API_VERSION)
    logger.info("=" * 60)

    # Step 1: Initialize database
    init_db()

    # Step 2: Initialize monitoring components
    app.state.faiss_indexer = FAISSNoveltyIndexer(n_neighbors=5, margin=1.5)
    app.state.drift_detector = DriftDetector(window_size=200)
    app.state.alert_engine = AlertEngine()
    app.state.fingerprint = None
    app.state.topology_cache = None
    app.state.inference_count = 0
    app.state.model_sha256 = None

    # Restore alerts that were still active when the server last stopped
    try:
        with SessionLocal() as db:
            restored = load_active_alerts(db)
        for alert in restored:
            app.state.alert_engine.restore(alert)
        if restored:
            logger.info("Restored %d active alert(s) from the database.", len(restored))
    except Exception:
        logger.error("Could not restore alerts from the database.", exc_info=True)

    if not settings.admin_api_key:
        logger.warning(
            "GUARDIAN_ADMIN_API_KEY is not set — admin endpoints "
            "(probe sweep, alert resolution) are disabled."
        )

    # Step 3: Load XGBoost model
    logger.info("Loading XGBoost survival model from %s ...", settings.model_path)
    try:
        model = xgb.Booster()
        model.load_model(str(settings.model_path))
        app.state.model = model
        app.state.model_sha256 = _file_sha256(settings.model_path)
        logger.info("Model loaded successfully (sha256 %s).", app.state.model_sha256)
    except Exception:
        logger.error("CRITICAL: Failed to load model.", exc_info=True)
        app.state.model = None

    # Step 4: Automatically run probe sweep to build baseline
    if app.state.model is not None:
        n = max(100, min(settings.probe_samples_on_startup, 2000))
        logger.info("Running automatic LHS probe sweep (%d samples)...", n)
        try:
            fp, _ = run_probe_sweep(app.state, n)
            logger.info(
                "Probe sweep complete. Fingerprint: %s | Novelty threshold: %.4f",
                fp.fingerprint_id,
                app.state.faiss_indexer.threshold or 0.0
            )
        except Exception:
            logger.error("Probe sweep failed — novelty scoring disabled.", exc_info=True)

    logger.info("Server ready. All monitoring systems active.")
    logger.info("=" * 60)

    yield  # ← server is running

    # Shutdown
    logger.info("Shutting down. Releasing resources...")
    app.state.model = None
    logger.info("Server stopped.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Data Center Guardian API",
    description=(
        "Enterprise predictive maintenance platform with XGBoost survival analysis, "
        "FAISS latent space novelty detection, KS/PSI drift monitoring, and an "
        "automated alert engine."
    ),
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if settings.enable_docs else None,
    redoc_url="/redoc" if settings.enable_docs else None,
    openapi_url="/openapi.json" if settings.enable_docs else None,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return validation errors without echoing the raw request body back."""
    errors = [
        {"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")}
        for e in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Never leak stack traces or internals to clients."""
    request_id = getattr(request.state, "request_id", None)
    logger.error("Unhandled error (request_id=%s)", request_id, exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error.", "request_id": request_id},
    )


# Middleware order: the last added runs first (outermost).
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["POST", "GET", "DELETE"],
    allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(inference_router)
app.include_router(drives_router)
app.include_router(monitoring_router)
app.include_router(autopsy_router)


# ---------------------------------------------------------------------------
# Legacy compatibility route — keeps old clients working without changes
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=SurvivalPrediction, tags=["legacy"], deprecated=True)
@limiter.limit(settings.rate_limit_predict)
def predict_legacy(request: Request, telemetry: DriveTelemetry, db: Session = Depends(get_db)):
    """
    Legacy endpoint for backward compatibility with the old frontend.
    Delegates to the shared inference pipeline and returns the original shape.
    """
    pred = run_inference(request.app.state, [telemetry], db)[0]
    return SurvivalPrediction(
        ttf_days=pred.ttf_days,
        rul_days=pred.rul_days,
        risk_level=pred.risk_level,
        log_time=pred.log_time,
    )


@app.get("/telemetry", tags=["meta"])
def get_telemetry():
    """Returns the latest dataset telemetry (legacy compat)."""
    try:
        with open(settings.telemetry_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return JSONResponse(
            status_code=404,
            content={"detail": "Telemetry file not found. Run ETL pipeline first."},
        )


@app.get("/healthz", tags=["meta"])
def healthz(request: Request):
    """Liveness/readiness probe for container orchestrators. 503 until the model is loaded."""
    ready = getattr(request.app.state, "model", None) is not None
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"status": "ok" if ready else "unavailable", "model_loaded": ready},
    )


@app.get("/api", tags=["meta"])
def root(request: Request):
    """API root — returns service metadata."""
    state = request.app.state
    fp = getattr(state, "fingerprint", None)
    return {
        "service": "Data Center Guardian API",
        "version": API_VERSION,
        "status": "operational" if getattr(state, "model", None) is not None else "degraded",
        "model_sha256": getattr(state, "model_sha256", None),
        "fingerprint_id": fp.fingerprint_id if fp else None,
        "admin_enabled": bool(settings.admin_api_key),
        "max_batch_size": settings.max_batch_size,
        "docs": "/docs" if settings.enable_docs else None,
    }


# Mount static frontend at root — MUST be last so API routes take priority.
if settings.frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=str(settings.frontend_dist), html=True), name="frontend")
else:
    logger.warning("Frontend dist not found at %s — serving API only.", settings.frontend_dist)

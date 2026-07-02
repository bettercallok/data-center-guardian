"""
api/main.py — Data Center Guardian API

Enterprise-grade predictive maintenance platform.
On startup, the lifespan handler:
  1. Loads the XGBoost survival model
  2. Initializes the FAISS novelty indexer, drift detector, and alert engine
  3. Automatically runs a 500-sample LHS probe sweep to build the baseline
  4. Initializes the SQLite database

All monitoring is then active from the first inference call.
"""
import logging
import json
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import xgboost as xgb

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.api.routes_inference import router as inference_router
from src.api.routes_monitoring import router as monitoring_router
from src.api.routes_autopsy import router as autopsy_router
from src.database.connection import init_db
from src.monitoring.faiss_indexer import FAISSNoveltyIndexer
from src.monitoring.drift_metrics import DriftDetector
from src.monitoring.alert_engine import AlertEngine

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Startup: load model → init monitoring → run probe sweep.
    Shutdown: clean release of all resources.
    """
    logger.info("=" * 60)
    logger.info("DATA CENTER GUARDIAN — STARTUP")
    logger.info("=" * 60)

    # Step 1: Initialize database
    init_db()

    # Step 2: Initialize monitoring components
    app.state.faiss_indexer = FAISSNoveltyIndexer(n_neighbors=5, margin=2.0)
    app.state.drift_detector = DriftDetector(window_size=200)
    app.state.alert_engine = AlertEngine()
    app.state.fingerprint = None
    app.state.inference_count = 0

    # Step 3: Load XGBoost model
    logger.info("Loading XGBoost survival model...")
    try:
        model = xgb.Booster()
        model.load_model("src/api/survival_model.json")
        app.state.model = model
        logger.info("Model loaded successfully.")
    except Exception:
        logger.error("CRITICAL: Failed to load model.", exc_info=True)
        app.state.model = None

    # Step 4: Automatically run probe sweep to build baseline
    if app.state.model is not None:
        logger.info("Running automatic LHS probe sweep (500 samples)...")
        try:
            from src.probing.lhs_sampler import lhs_sample
            from src.probing.fingerprint import compile_fingerprint

            probe_X = lhs_sample(500)
            dmatrix = xgb.DMatrix(probe_X)
            probe_preds = app.state.model.predict(dmatrix)

            # Build FAISS index
            app.state.faiss_indexer.build(app.state.model, probe_X)

            # Calibrate drift detector
            app.state.drift_detector.calibrate(probe_X)

            # Compile fingerprint
            fp = compile_fingerprint(app.state.model, probe_X, probe_preds)
            app.state.fingerprint = fp

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

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Data Center Guardian API",
    description=(
        "Enterprise predictive maintenance platform with XGBoost survival analysis, "
        "FAISS latent space novelty detection, KS/PSI drift monitoring, and an "
        "automated alert engine."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — restrict to known trusted origins only
ALLOWED_ORIGINS = [
    "https://data-center-gaurdian.vercel.app",
    "https://bettercallok-data-center-guardian.hf.space",
    "http://localhost:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "GET", "DELETE"],
    allow_headers=["Content-Type"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(inference_router)
app.include_router(monitoring_router)
app.include_router(autopsy_router)


# ---------------------------------------------------------------------------
# Legacy compatibility route — keeps old frontend working without changes
# ---------------------------------------------------------------------------
@app.post("/predict")
async def predict_legacy(request):
    """
    Legacy endpoint for backward compatibility with the old frontend.
    Delegates to the new enriched inference route.
    """
    from fastapi import Request as FRequest
    from src.api.schemas import DriveTelemetry
    body = await request.json()
    tel = DriveTelemetry(**body)

    # Re-use the new inference logic
    from src.api.routes_inference import predict as _predict
    return await _predict(request, tel)


@app.get("/telemetry")
async def get_telemetry():
    """Returns the latest dataset telemetry (legacy compat)."""
    try:
        with open("data/processed/telemetry.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": "Telemetry file not found. Run ETL pipeline first."}


@app.get("/api")
async def root():
    """API root — returns service metadata."""
    return {
        "service": "Data Center Guardian API",
        "version": "2.0.0",
        "status": "operational",
        "docs": "/docs",
    }


# Mount static frontend at root — MUST be last so API routes take priority.
# All /api/... routes are registered above and will match before this catch-all.
try:
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")
except Exception:
    logger.warning("Frontend dist not found — serving API only.")

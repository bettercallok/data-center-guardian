# ─── Stage 1: Build React Frontend ────────────────────────────────────────────
FROM node:20-slim AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ .
RUN npm run build

# ─── Stage 2: Python Backend ───────────────────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create a non-root user (required by Hugging Face Spaces, good practice everywhere)
RUN useradd -m -u 1000 user

WORKDIR /app

# Install system-level dependencies needed by faiss-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all backend source modules
COPY src/ src/

# Copy data
COPY data/processed/ data/processed/

# Copy built frontend from Stage 1
COPY --from=frontend-builder /frontend/dist frontend/dist

# Create the SQLite database directory and set permissions
RUN mkdir -p /app/data && chown -R user:user /app

# Switch to non-root user
USER user
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

# Database defaults to SQLite inside the container
ENV DATABASE_URL=sqlite:////app/guardian.db

# Admin endpoints (probe sweep, alert resolution) stay disabled unless
# GUARDIAN_ADMIN_API_KEY is provided at runtime (e.g. as a HF Space secret).
# Never bake secrets into the image.

# Expose the app port
EXPOSE 7860

# Container health: 200 once the model is loaded
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/healthz', timeout=4).status == 200 else 1)"

# Run FastAPI via Uvicorn. --proxy-headers lets the rate limiter see real
# client IPs behind the Hugging Face / load-balancer proxy; restrict which
# upstreams are trusted with FORWARDED_ALLOW_IPS when not behind a proxy.
CMD ["sh", "-c", "exec uvicorn src.api.main:app --host 0.0.0.0 --port 7860 --proxy-headers --forwarded-allow-ips=\"${FORWARDED_ALLOW_IPS:-*}\" --no-server-header"]

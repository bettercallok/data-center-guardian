# ─── Stage 1: Build React Frontend ────────────────────────────────────────────
FROM node:20-slim AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ .
RUN npm run build

# ─── Stage 2: Python Backend ───────────────────────────────────────────────────
FROM python:3.11-slim

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

# Expose the app port
EXPOSE 7860

# Run FastAPI via Uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]

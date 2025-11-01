# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers

# Install basic system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only dependency files first for layer caching
COPY pyproject.toml /app/
COPY README.md /app/

# Optional: install build tools (setuptools, wheel)
RUN pip install --no-cache-dir setuptools wheel

# CPU-only torch (faster build)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.3.1

# Install your project + dependencies
COPY . /app
RUN pip install --no-cache-dir .

# Set Python path for internal imports
ENV PYTHONPATH=/app

# Expose FastAPI port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# Run the app
CMD ["sh", "-c", "uvicorn rag.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-serve.txt ./
RUN python -m pip install --only-binary=:all: -r requirements-serve.txt

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home app \
    && mkdir -p /app/artifacts /app/data/features /app/data/curated \
    && chown -R app:app /app
COPY --chown=app:app src/ ./src/
COPY --chown=app:app configs/config.yaml ./configs/config.yaml
USER app
EXPOSE 8010

# Readiness requires an actually loaded scoring model, not merely HTTP 200.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import json,urllib.request; r=json.load(urllib.request.urlopen('http://127.0.0.1:8010/health',timeout=3)); raise SystemExit(0 if r.get('status')=='ok' else 1)"
CMD ["python", "-m", "uvicorn", "jaga_claim.api.app:app", "--host", "0.0.0.0", "--port", "8010", "--workers", "1"]

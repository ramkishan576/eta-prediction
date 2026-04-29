# ── Base image ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code and artefacts ────────────────────────────────────────────
COPY src/        src/
COPY predict.py  .
COPY artifacts/  artifacts/

# ── Inference entry point ──────────────────────────────────────────────────────
# Expected invocation:
#   docker run --rm \
#     -v /path/to/test.csv:/data/test.csv \
#     -v /path/to/output:/output \
#     eta-prediction \
#     --input /data/test.csv --output /output/predictions.csv

ENTRYPOINT ["python", "predict.py"]

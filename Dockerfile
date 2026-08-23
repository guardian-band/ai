FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POLYPHARMACY_DEVICE=cpu \
    POLYPHARMACY_CPU_THREADS=1 \
    POLYPHARMACY_EXPERIMENT_PATH=/models/teacher.yaml \
    POLYPHARMACY_MANIFEST_PATH=/models/benchmark/manifest.json \
    POLYPHARMACY_CHECKPOINT_PATH=/models/checkpoint_best.pt \
    POLYPHARMACY_SELECTION_PATH=/models/teacher_validation_selection.json

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r requirements-api.txt

COPY . .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=150s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3)"

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# Polypharmacy AI API deployment

The service loads the frozen warm-pair Morgan+MPNN Teacher once at process
startup. Subsequent requests reuse the in-memory model. Unknown DrugBank IDs are
rejected rather than receiving an unsupported confident prediction.

## 1. Create the model release directory

```bash
python scripts/package_teacher_api_release.py \
  --experiment /path/to/teacher.yaml \
  --manifest /path/to/warm_pair/seed_42/manifest.json \
  --checkpoint /path/to/checkpoint_best.pt \
  --selection /path/to/teacher_validation_selection.json \
  --output model-release
```

The generated directory contains the model, features, hierarchy, benchmark
contract and their hashes. It is excluded from the Docker image and mounted
read-only at runtime.

## 2. Start locally without Docker

```bash
export POLYPHARMACY_EXPERIMENT_PATH="$PWD/model-release/teacher.yaml"
export POLYPHARMACY_MANIFEST_PATH="$PWD/model-release/benchmark/manifest.json"
export POLYPHARMACY_CHECKPOINT_PATH="$PWD/model-release/checkpoint_best.pt"
export POLYPHARMACY_SELECTION_PATH="$PWD/model-release/teacher_validation_selection.json"
export POLYPHARMACY_DEVICE=cpu
uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --workers 1
```

Open `http://127.0.0.1:8000/docs` for the generated API documentation.

## 3. Start with Docker

```bash
docker compose up --build
```

The first `/ready` success can take tens of seconds because source hashes,
features and the checkpoint are verified and loaded. This happens once per
container process, not once per prediction. Keep one worker per container and
scale by adding replicas so every process does not duplicate the model.

The production image installs the official CPU-only PyTorch wheel. The frozen
Teacher does not require CUDA inside the API container. On the tested Apple
Silicon host the container became ready in about 17 seconds; startup time will
vary by storage and CPU. The first prediction may include framework warm-up,
while subsequent requests reuse the already loaded model.

## Endpoints

- `GET /health`: process health.
- `GET /ready`: model readiness and immutable checkpoint identity.
- `GET /v1/model`: model metadata and the 15 Level-1 organ labels.
- `POST /v1/predict`: Level-1 organ and Level-2 side-effect top-k predictions.

Example request:

```json
{
  "drug_a": "DB00313",
  "drug_b": "DB01041",
  "top_k": 5
}
```

This API is a decision-support prototype, not medical advice. It supports drugs
present in the frozen feature artifact. Raw-SMILES/novel-drug inference is not
part of this release.

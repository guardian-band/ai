# Polypharmacy AI API deployment

The service loads the frozen warm-pair Morgan+MPNN Teacher once at process
startup. The production default is ONNX Runtime on CPU; the verified PyTorch
runtime remains available as a fallback. Both runtimes preserve the fused
Teacher and Morgan-only fallback behavior. Subsequent requests reuse the
in-memory sessions. Unknown DrugBank IDs are rejected rather than receiving an
unsupported confident prediction.

## 1. Create the model release directory

```bash
python scripts/package_teacher_api_release.py \
  --experiment /path/to/teacher.yaml \
  --manifest /path/to/warm_pair/seed_42/manifest.json \
  --checkpoint /path/to/checkpoint_best.pt \
  --selection /path/to/teacher_validation_selection.json \
  --onnx-dir /path/to/verified/onnx \
  --output model-release
```

The generated directory contains the model, features, hierarchy, benchmark
contract and their hashes. It is excluded from the Docker image and mounted
read-only at runtime.

The ONNX directory is produced and verified before packaging:

```bash
python scripts/export_frozen_teacher_onnx.py \
  --experiment /path/to/teacher.yaml \
  --manifest /path/to/manifest.json \
  --checkpoint /path/to/checkpoint_best.pt \
  --selection /path/to/teacher_validation_selection.json \
  --output /path/to/verified/onnx

python scripts/validate_onnx_teacher.py \
  --experiment /path/to/teacher.yaml \
  --manifest /path/to/manifest.json \
  --checkpoint /path/to/checkpoint_best.pt \
  --selection /path/to/teacher_validation_selection.json \
  --onnx-dir /path/to/verified/onnx \
  --output /path/to/verified/onnx/onnx_validation_benchmark.json
```

## 2. Start locally without Docker

```bash
export POLYPHARMACY_EXPERIMENT_PATH="$PWD/model-release/teacher.yaml"
export POLYPHARMACY_MANIFEST_PATH="$PWD/model-release/benchmark/manifest.json"
export POLYPHARMACY_CHECKPOINT_PATH="$PWD/model-release/checkpoint_best.pt"
export POLYPHARMACY_SELECTION_PATH="$PWD/model-release/teacher_validation_selection.json"
export POLYPHARMACY_RUNTIME=onnx
export POLYPHARMACY_ONNX_FUSED_PATH="$PWD/model-release/onnx/teacher_fused.onnx"
export POLYPHARMACY_ONNX_BASELINE_PATH="$PWD/model-release/onnx/teacher_baseline.onnx"
export POLYPHARMACY_ONNX_RELEASE_PATH="$PWD/model-release/onnx/onnx_release.json"
export POLYPHARMACY_DEVICE=cpu
uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --workers 1
```

Open `http://127.0.0.1:8000/docs` for the generated API documentation.

## 3. Start with Docker

```bash
docker compose up --build
```

The first `/ready` success includes source-hash verification, feature loading and
ONNX session creation. This happens once per container process, not once per
prediction. Keep one worker per container and scale by adding replicas.

The production image retains the official CPU-only PyTorch wheel as a verified
fallback, but inference defaults to ONNX Runtime. The frozen Teacher does not
require CUDA inside the API container. To force the original backend, set
`POLYPHARMACY_RUNTIME=pytorch` and omit the three ONNX path variables. The first
prediction may include runtime warm-up, while subsequent requests reuse the
already loaded model.

Native Apple Silicon validation produced exact Top-5 parity on 16 validation
pairs. Maximum ONNX-versus-PyTorch probability error was approximately
`1.1e-6`. One-thread ONNX batch-1 latency was about `1.37 ms` for fused Teacher
inference and `0.075 ms` for Morgan fallback. These are host-specific reference
measurements; container and production-host latency must be measured separately.

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

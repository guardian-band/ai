# Polypharmacy side-effect prediction

This repository implements leakage-checked multi-label prediction for adverse
effects of drug pairs. It includes warm-pair, cold-1, and cold-2 benchmarks;
simple baselines; and an offline multimodal teacher distilled into a CPU-safe
single-pair student.

No accuracy, AUROC, or average-precision result is hard-coded or promised.
Results are valid only when they come from completed, hash-verified run
artifacts and the strict aggregator.

## Model paths

- `prevalence`, `logistic`, and `symmetric_mlp` are baseline models.
- `multimodal_teacher` combines Morgan fingerprints, pinned MolFormer token
  embeddings, a self-supervised molecular MPNN, and a typed PrimeKG HGT.
- `distilled_pair_student` uses the teacher's cached latent drug tokens and is
  the production CPU inference model.

The advanced architecture, artifact contracts, exact commands, and cold-drug
graph protocol are documented in
[`docs/MULTIMODAL_TEACHER_STUDENT.md`](docs/MULTIMODAL_TEACHER_STUDENT.md).

## Install

Python 3.10+ is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PrimeKG HGT training uses PyG neighbor sampling. GPU environments may also
need the PyG wheels matching their exact PyTorch/CUDA versions, as described
by the official PyG installation guide.

## Baseline experiment

Build benchmark manifests, then run one declared model configuration:

```bash
python build_benchmark.py --config configs/benchmark.yaml
python run_experiment.py \
  --config configs/models/symmetric_mlp.yaml \
  --benchmark artifacts/benchmarks/polypharmacy_v1/warm_pair/seed_42/manifest.json
```

Run every discovered manifest only after preflight succeeds:

```bash
python run_experiment.py \
  --config configs/models/symmetric_mlp.yaml \
  --all-benchmarks \
  --benchmarks-root artifacts/benchmarks/polypharmacy_v1
```

## Advanced experiment

Use the individual producer commands or the resumable orchestrator:

```bash
python run_advanced_pipeline.py \
  --config configs/advanced_pipeline.example.yaml \
  --dry-run
```

Replace every placeholder in the example before execution. Real training and
model downloads are intentionally never triggered by tests.

## Verification

```bash
python -m pytest -q
```

The report generator consumes only strict aggregate JSON. It does not invent
metrics when runs or artifacts are missing.

# Advanced multimodal teacher and CPU student

## What is implemented

The advanced path is executable end to end. It contains:

1. pinned MolFormer token extraction from canonical SMILES;
2. a bond-aware molecular MPNN and masked-atom self-supervised trainer;
3. typed PrimeKG preprocessing with target-leakage relation removal;
4. an HGT-style relation-aware attention encoder, link pretraining, neighbor
   sampling, validation selection, and layerwise drug-token export;
5. a versioned four-modality artifact assembler with exact manifest binding;
6. the symmetric cross-attention teacher and hierarchical organ/specific heads;
7. teacher latent-token cache export;
8. supervised plus logit-distilled CPU student training;
9. calibrated, thresholded, single-pair CPU inference.

The repository does not contain downloaded MolFormer weights, trained MPNN or
HGT checkpoints, feature artifacts, or benchmark results. Those are outputs of
the commands below. Their absence does not mean the producer code is missing.

## Architecture

For each drug, the teacher consumes four independent modalities:

- Morgan fingerprint: `[2048]`;
- eight mask-aware segment-pooled MolFormer hidden-state tokens: `[8, 768]`;
- molecular-MPNN atom tokens: `[32, 256]` by default;
- PrimeKG-HGT layer tokens: `[layers + 1, 256]` by default.

Each sequence has an availability flag and a real padding mask. Padding is
excluded from the learned resampler. A missing sequence contributes one
learned missing-modality token rather than a repeated padded signal.

The resampler converts the modalities into eight learned `128`-dimensional
latent tokens per drug. The pair decoder performs token cross-attention in
both directions. The public output averages the `A→B` and `B→A` ordered
passes, making the organ and specific-side-effect predictions exactly
invariant to pair order.

The CPU student consumes only the cached `[8, 128]` tokens. It does not import
RDKit, Transformers, PrimeKG, or PyG at inference time.

## Leakage rules

The PrimeKG producer removes relations that directly encode the prediction
target, including drug-drug, drug-effect, indication, contraindication,
off-label-use, and synergistic-interaction relations.

For every remaining typed relation, message-passing edges, train link labels,
and validation link labels are disjoint. A held-out edge and its generated
reverse edge are both absent from the message graph. Negative link labels are
rejected against the full known positive relation, not sampled blindly.

For cold-1 and cold-2, every benchmark drug absent from the downstream train
split is removed—together with all incident safe KG edges—from HGT pretraining.
The frozen HGT is then applied to the complete safe KG during export. Thus the
protocol is encoder-train inductive: no cold drug shapes HGT weights, while
safe biological neighbors available at inference can still describe it. The
artifact records this protocol and the excluded-drug count.

## Artifact chain

All token and multimodal artifacts are safe NPZ containers loaded with
`allow_pickle=False`. They require sorted unique drug IDs, finite float32
arrays, boolean masks, canonical metadata, an internal content digest, and an
adjacent file SHA-256. The multimodal artifact is bound to the exact
`benchmark_id`, scenario, seed, and manifest hash.

The provenance chain is:

```text
drug table + pinned MolFormer revision ──> MolFormer tokens
drug table + selected MPNN checkpoint ──> MPNN tokens
PrimeKG + Morgan + manifest + HGT checkpoint ──> HGT tokens
all four modalities + exact manifest ──> multimodal feature artifact
multimodal artifact + selected teacher checkpoint ──> latent-token cache
cache + teacher checkpoint/config ──> distilled student checkpoint
student + cache + calibration + thresholds ──> CPU pair prediction
```

## Exact execution order

The following commands show one manifest. Repeat the HGT, assembly, teacher,
cache, and student stages for each scenario and seed.

### 1. MolFormer tokens

The official model uses custom Hugging Face code. Remote code must be opted in
explicitly and the revision must be a 40-character commit hash, never `main`.

```bash
python scripts/extract_molformer_tokens.py \
  --input data/raw/drugs_master.csv \
  --output artifacts/features/molformer_tokens.npz \
  --revision REPLACE_WITH_40_CHARACTER_COMMIT_HASH \
  --allow-remote-code \
  --device cuda
```

### 2. Molecular MPNN

```bash
python scripts/train_molecular_mpnn.py \
  --input data/raw/drugs_master.csv \
  --checkpoint artifacts/checkpoints/molecular_mpnn.pt \
  --device cuda

python scripts/export_molecular_mpnn_tokens.py \
  --input data/raw/drugs_master.csv \
  --checkpoint artifacts/checkpoints/molecular_mpnn.pt \
  --output artifacts/features/mpnn_tokens.npz \
  --device cuda
```

### 3. Typed PrimeKG HGT

```bash
python scripts/train_primekg_hgt.py \
  --primekg data/external/primekg_kg.csv \
  --morgan artifacts/morgan_fingerprints.parquet \
  --manifest artifacts/benchmarks/polypharmacy_v1/cold_1/seed_42/manifest.json \
  --checkpoint artifacts/checkpoints/hgt_cold_1_seed_42.pt \
  --output artifacts/features/hgt_cold_1_seed_42.npz \
  --device cuda
```

### 4. Assemble one manifest-bound artifact

```bash
python build_advanced_features.py \
  --manifest artifacts/benchmarks/polypharmacy_v1/cold_1/seed_42/manifest.json \
  --morgan artifacts/morgan_fingerprints.parquet \
  --molformer artifacts/features/molformer_tokens.npz \
  --mpnn artifacts/features/mpnn_tokens.npz \
  --kg artifacts/features/hgt_cold_1_seed_42.npz \
  --output artifacts/features/advanced_cold_1_seed_42.npz
```

### 5. Materialize and train the teacher

This command measures dimensions and writes every required path and SHA-256;
do not edit hashes by hand.

```bash
python configure_advanced_experiment.py \
  --model teacher \
  --template configs/model_multimodal_teacher.yaml \
  --manifest artifacts/benchmarks/polypharmacy_v1/cold_1/seed_42/manifest.json \
  --features artifacts/features/advanced_cold_1_seed_42.npz \
  --hierarchy artifacts/meddra_hierarchy.json \
  --output artifacts/configs/teacher_cold_1_seed_42.yaml

python run_precomputed_experiment.py \
  --experiment artifacts/configs/teacher_cold_1_seed_42.yaml \
  --benchmark artifacts/benchmarks/polypharmacy_v1/cold_1/seed_42/manifest.json
```

### 6. Cache the selected teacher

```bash
python build_teacher_cache.py \
  --experiment artifacts/configs/teacher_cold_1_seed_42.yaml \
  --benchmark artifacts/benchmarks/polypharmacy_v1/cold_1/seed_42/manifest.json \
  --output artifacts/features/teacher_cache_cold_1_seed_42.npz \
  --device cpu
```

### 7. Materialize and train the student

```bash
python configure_advanced_experiment.py \
  --model student \
  --template configs/model_distilled_pair_student.yaml \
  --manifest artifacts/benchmarks/polypharmacy_v1/cold_1/seed_42/manifest.json \
  --cache artifacts/features/teacher_cache_cold_1_seed_42.npz \
  --teacher-config artifacts/configs/teacher_cold_1_seed_42.yaml \
  --hierarchy artifacts/meddra_hierarchy.json \
  --output artifacts/configs/student_cold_1_seed_42.yaml

python run_precomputed_experiment.py \
  --experiment artifacts/configs/student_cold_1_seed_42.yaml \
  --benchmark artifacts/benchmarks/polypharmacy_v1/cold_1/seed_42/manifest.json
```

Both training runs select checkpoints only on validation macro average
precision. Calibration and thresholds are fit only after model selection, on
validation logits. Test construction remains blocked until validation is
frozen.

## Resumable orchestration

`run_advanced_pipeline.py` executes only the allow-listed stages declared in
`configs/advanced_pipeline.example.yaml`. A stage accepts one argument list or
multiple argument lists, which is how scenario/seed jobs are expressed.

```bash
python run_advanced_pipeline.py \
  --config configs/advanced_pipeline.example.yaml \
  --dry-run

python run_advanced_pipeline.py \
  --config configs/advanced_pipeline.example.yaml \
  --start-at hgt \
  --stop-after student
```

## Performance claims

This implementation makes no performance guarantee. The architecture is an
advanced candidate, not a measured result. Report warm-pair, cold-1, and
cold-2 macro/micro average precision, AUROC as a secondary metric, ranking
metrics, calibration, and five-seed uncertainty only from completed,
hash-verified runs.

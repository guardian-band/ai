# Multimodal teacher and CPU student

This repository now contains the model and artifact contracts for a two-stage
deployment design. It does not contain MolFormer, a molecular MPNN, an HGT,
PrimeKG preprocessing, pretrained weights, or training outputs.

## Offline teacher boundary

`MultiModalTeacher` consumes three validated per-drug inputs:

- a Morgan/global chemistry vector;
- fixed-count molecular tokens;
- fixed-count PrimeKG/biological tokens.

Each modality has an availability mask. The teacher projects the modalities to
a shared latent space, performs token-level cross-drug attention, and uses
learned organ and specific-side-effect queries. It emits separate organ and
specific logits. The public pair function averages both ordered interaction
passes, so `f(A, B)` and `f(B, A)` are the same model output.

The upstream producers must provide finite tensors and explicit provenance
hashes. `run_precomputed_experiment.py` validates those artifacts and every
manifest split before creating run artifacts; the existing legacy manifest
runner intentionally rejects these model types because its dataset emits only
the legacy single feature tensor.

## CPU student boundary

`DistilledPairStudent` consumes only a fixed `[token_count, token_dim]` latent
token matrix per drug plus token availability. It has no PyG or Transformers
runtime dependency. `CPUPairPredictor` loads a validated
`CachedTokenArtifact`, requires the student to already be in `eval()` mode on
CPU, looks up one unordered pair, applies stored temperatures once to logits,
and returns calibrated probabilities, thresholded predictions, and
deterministically ordered Top-K results.

The cache format is a versioned `.npz` container with `allow_pickle=False`, a
canonical payload checksum, and an adjacent file checksum. Writes are atomic.
Drug IDs are sorted and unique; token arrays are finite float32 values with a
fixed count and dimension. Metadata records canonical SHA-256 hashes for the
teacher checkpoint, teacher configuration, and modality provenance. Missing
IDs and tampered artifacts are rejected.

## Known drugs and new drugs

Known-drug pair inference requires both IDs in the cache and does not load
PrimeKG or MolFormer. A new SMILES is not silently supported by the CPU pair
predictor: it must go through a separately provisioned offline feature
producer, then be added to a newly validated cache. If no biological modality
is available, the teacher's availability-mask path can still produce finite
outputs, but that behavior must be measured rather than assumed to improve
cold-drug performance.

## Required artifacts before training/deployment

Before a real run, provide:

1. finite, shape-validated Morgan, molecular-token, and KG-token artifacts;
2. provenance hashes for each upstream modality and the exact multimodal feature artifact file;
3. a versioned side-effect organ mapping and validation-selected calibration;
4. a distilled student checkpoint matching the cache token schema;
5. a `CachedTokenArtifact` plus its `.sha256` sidecar. Its
   `modality_provenance_hash` must equal the SHA-256 of the exact multimodal
   feature artifact file used to produce the cache.

The teacher YAML requires `feature_artifact_path`,
`feature_artifact_sha256`, `hierarchy_path`, and `hierarchy_sha256`, in
addition to the explicit model and training controls. The student YAML
requires `cached_token_artifact_path`, `cached_token_artifact_sha256`,
`teacher_config_path`, `teacher_config_sha256`, `teacher_checkpoint_path`,
`teacher_checkpoint_sha256`, `hierarchy_path`, and `hierarchy_sha256`, plus
`supervised_weight` and `distillation_weight`. The feature artifact
compatibility record must contain the exact `benchmark_id`, `scenario`,
`seed`, and `manifest_hash`.

No percentage or benchmark result is implied by these code paths. Training and
benchmark generation remain separate operations and were not run as part of
this implementation.

## Staged execution

After the upstream producers have supplied real artifacts and their SHA-256
values have been written into the YAML files, run preflight first:

```bash
python run_precomputed_experiment.py \
  --experiment configs/model_multimodal_teacher.yaml \
  --benchmark artifacts/benchmarks/polypharmacy_v1/warm_pair/seed_42/manifest.json \
  --dry-run
```

Then run the teacher, create and validate the cached-token artifact, and place
its path/hash plus the frozen teacher checkpoint/config path/hash in
`configs/model_distilled_pair_student.yaml`. Preflight the student before its
actual run:

```bash
python run_precomputed_experiment.py \
  --experiment configs/model_distilled_pair_student.yaml \
  --benchmark artifacts/benchmarks/polypharmacy_v1/warm_pair/seed_42/manifest.json \
  --dry-run
```

The existing legacy manifest runner is intentionally not used for these model
types. A separate invocation is required for each manifest/seed; aggregation
must continue through the repository's strict `aggregate_results.py` path.

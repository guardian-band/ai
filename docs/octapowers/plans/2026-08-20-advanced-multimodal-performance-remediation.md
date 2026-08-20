# Advanced Multimodal Performance Remediation Plan

**Date:** 2026-08-20  
**Target branch:** `work/advanced-multimodal-teacher-student`  
**Goal:** Make the advanced Teacher–Student pipeline capable of adding graph and molecular information without discarding the established Morgan baseline, then demonstrate the gain with leakage-safe, matched experiments.

## Non-negotiable constraints

- Do not run or report a test set while choosing architecture, loss weights, gates, thresholds, calibration, or checkpoints.
- Compare every model on the same v2 manifest, ordered label vocabulary, negative controls, and seed.
- Use macro AUPRC as the primary checkpoint and promotion metric. Keep micro AUPRC, macro/micro AUROC, Brier, ECE, and per-label AP as secondary metrics.
- Preserve pair-order invariance.
- Preserve artifact SHA-256 provenance and validation-before-test state guards.
- Do not claim that the advanced model improves performance until the paired multi-seed confidence interval supports that conclusion.
- Do not start the five-seed Teacher–Student sweep until Tasks 1–6 pass their acceptance gates.

## Task 0 — Freeze the current evidence and establish matched controls

**Problem addressed:** The current advanced result is not directly comparable with the earlier Morgan result.

**Files:**

- Add `configs/ablations/` with v2 experiment configurations.
- Extend `aggregate_results.py` only if it cannot already group the new model identifiers.
- Add or extend aggregation tests in `tests/test_aggregation_provenance.py`.

**Implementation:**

1. Preserve the existing seed-42 Teacher and Student artifacts as immutable evidence. Record their run IDs, manifest hash, label hash, feature hashes, and code commit.
2. Create matched configurations for:
   - `morgan_only`
   - `morgan_molformer`
   - `morgan_mpnn`
   - `morgan_hgt`
   - `full_multimodal_teacher`
   - `distilled_pair_student`
3. Every configuration must point to the same v2 benchmark manifest and hierarchy artifact. The only permitted difference is the enabled modality set/model implementation.
4. Add the enabled modality list to `config.resolved.json` and the run identity. Reject aggregation if supposedly paired runs have different manifest hashes or label hashes.
5. Do not reuse the old v1 Morgan metric as the control.

**Acceptance:**

- A dry run validates all six configurations against one v2 seed without creating run artifacts.
- Aggregation rejects a comparison when manifest hash, label order/hash, scenario, or seed differs.
- A seed-42 Morgan-only v2 run exists before evaluating whether multimodal fusion helped.

## Task 1 — Repair HGT link-prediction optimization

**Problem addressed:** Mean validation BCE of 7–12 indicates severely mis-scaled link logits; decreasing loss alone does not validate the graph embeddings.

**Files:**

- `src/models/primekg_hgt.py`
- `scripts/train_primekg_hgt.py`
- `tests/models/test_primekg_hgt.py`
- Add `tests/test_primekg_hgt_training.py`

**Implementation:**

1. Replace the unscaled 256-dimensional bilinear sum in `TypedLinkPredictor` with a scaled relation-specific score:

   ```python
   score = (source * relation * destination).sum(dim=-1) / sqrt(hidden_dim) + relation_bias
   ```

   Add one learned scalar bias per relation. Keep relation vectors initialized to ones so initial logit variance is controlled by the `sqrt(hidden_dim)` divisor.
2. Make the HGT training function return, for every relation and globally:
   - mean BCE;
   - AP;
   - AUROC when both classes are present;
   - positive-logit mean/std;
   - negative-logit mean/std;
   - positive and negative counts.
3. Compute macro relation AP by giving each relation equal weight. Continue reporting micro metrics separately; do not let the largest PrimeKG relation determine checkpoint selection.
4. Select the HGT checkpoint by highest validation macro relation AP. Use lower validation BCE and then earlier epoch as deterministic tie-breakers.
5. Keep validation negatives fixed across epochs. Continue resampling training negatives by epoch.
6. Persist `hgt_training_metrics.json` next to the checkpoint and include its SHA-256 in pipeline stage provenance.
7. Log non-finite values and fail immediately if any loss, logit, embedding, or metric input is non-finite.

**Tests:**

- A deterministic synthetic separable graph must obtain validation AP greater than `0.90`.
- Initial logits for normalized fixture inputs must remain finite and have bounded scale; the test must fail against the old unscaled predictor.
- Checkpoint selection must prefer macro relation AP rather than edge-count-weighted BCE.
- Relation metrics must preserve undefined AUROC as `null`, not a fabricated number.

**Acceptance:**

- On the real warm-pair seed-42 HGT run, validation loss must be interpretable and finite; a value above `2.0` requires stopping and investigating logits rather than exporting tokens.
- Export is permitted only when validation macro relation AP exceeds the random/negative-sampling baseline and positive logits rank above negative logits.
- Do not set a universal AP target beyond this gate until real relation prevalences are inspected.

## Task 2 — Make HGT export genuinely inductive and distribution-consistent

**Problem addressed:** Cold-drug edges are removed during training but the current export reconstructs the complete PrimeKG graph, reintroducing excluded cold-drug neighborhoods.

**Files:**

- `scripts/train_primekg_hgt.py`
- `src/data/primekg_typed_graph.py`
- `tests/test_primekg_hgt_training.py`
- `tests/data/test_precomputed_artifacts.py`

**Implementation:**

1. Export HGT tokens from the same message-passing edge set used to train the selected encoder. Do not rebuild `full_data` from `prepared.edge_index` after checkpoint selection.
2. For `cold_1` and `cold_2`, keep excluded drugs as isolated nodes with their permitted non-KG input features, but do not restore any incident PrimeKG edge.
3. Keep HGT validation supervision edges excluded from the export graph. They were used for model selection and must not become inference-time message edges.
4. Persist these fields in the HGT artifact metadata:
   - message-graph SHA-256;
   - excluded drug IDs SHA-256 and count;
   - per-relation message/train/validation edge counts;
   - number and percentage of benchmark drugs with at least one permitted KG neighbor;
   - protocol value `train_message_graph_only`.
5. Remove the current misleading `label_and_encoder_train_inductive_safe_graph_at_export` value.

**Tests:**

- Adding or removing an edge incident to an excluded cold drug in the source graph must not change that cold run’s exported tokens.
- No exported cold graph may contain an edge whose source or destination is an excluded drug.
- Warm export must use the recorded message graph, not hidden validation edges.
- Artifact loading must reject a graph hash that differs from the declared message graph.

**Acceptance:**

- Automated tests prove edge exclusion at both training and export boundaries.
- Cold artifact metadata reports coverage rather than silently presenting isolated-drug tokens as fully graph-informed tokens.

## Task 3 — Preserve information in MolFormer and MPNN artifacts

**Problem addressed:** MolFormer uses arbitrary eight-bin averaging, while MPNN silently discards atoms after the configured token limit.

**Files:**

- `src/features/molformer_token_producer.py`
- `src/models/molecular_mpnn.py`
- `src/features/token_feature_artifact.py`
- `configs/model_multimodal_teacher.yaml`
- `tests/features/test_molformer_token_producer.py`
- `tests/models/test_molecular_mpnn.py`

**Implementation:**

1. MolFormer artifact generation must preserve the contextual hidden states and their true padding mask up to the configured tokenizer `max_length`. Remove consecutive-bin mean pooling.
2. Set the artifact’s MolFormer token count from the producer output and validate it against the Teacher config during preflight. Do not silently reshape or average to satisfy the config.
3. MPNN export must not silently truncate molecules. Before export, calculate the maximum atom count over required benchmark drugs:
   - if `token_count` covers it, export all atom tokens with padding;
   - otherwise fail with the required minimum token count and affected drug IDs.
4. Record token-count distributions, truncation count (required to be zero), model checkpoint hash, source hash, and upstream validation loss in artifact metadata.
5. Keep upstream encoders frozen for the first repaired ablation. Do not add expensive fine-tuning until the frozen-feature contribution is measured.
6. If the matched ablation shows that MolFormer or MPNN does not improve validation macro AUPRC, disable that modality in the promoted model. Do not retain a modality merely because it sounds advanced.

**Tests:**

- MolFormer output tokens at valid positions must equal the model’s contextual hidden states; padding positions must remain masked.
- A molecule longer than the configured MPNN token count must produce a clear failure, not truncated output.
- Round-trip artifacts must preserve masks, token values, dimensions, and provenance.

**Acceptance:**

- No producer performs undocumented lossy pooling or truncation.
- Every enabled molecular modality must show a non-negative validation contribution in a matched ablation before the final sweep.

## Task 4 — Replace token-count-biased fusion with balanced modality encoders

**Problem addressed:** One Morgan token currently competes directly with 8 MolFormer, 32 MPNN, and 4 KG tokens, with no per-modality normalization or reliability control.

**Files:**

- `src/models/multimodal_teacher_student.py`
- `src/models/factory.py`
- `configs/model_multimodal_teacher.yaml`
- `tests/models/test_multimodal_teacher_student.py`
- `tests/models/test_model_factory.py`

**Implementation:**

1. Give each modality its own `Linear -> LayerNorm` projection into `hidden_dim`.
2. Add an independent learned-query resampler for MolFormer, MPNN, and HGT. Each resampler must emit the same configurable number of summary tokens, regardless of raw modality token count.
3. Keep Morgan out of the auxiliary token competition; it will have the direct baseline path described in Task 5.
4. Add one learned availability/reliability gate per auxiliary modality and output head. Initialize gate logits to `-4.0`, so auxiliary corrections begin near zero.
5. Apply modality dropout only during training, independently dropping complete auxiliary modalities while never dropping Morgan. Add a single explicit config value `modality_dropout`; validate it in `[0, 1)`.
6. Expose detached gate values and modality availability counts through training diagnostics; do not change the model’s prediction return type.

**Tests:**

- Changing raw token count while representing the same padded content must not change the number of resampled tokens.
- Missing or completely padded modalities must produce finite outputs.
- Initial auxiliary gates must be near zero.
- Swapping drug A and B must preserve organ and specific logits.
- Gradients must reach every enabled modality projection and resampler.

**Acceptance:**

- Every auxiliary modality contributes the same number of normalized summary tokens.
- Attention allocation cannot be dominated solely by the original token count.
- Training artifacts contain final gate values and per-modality coverage.

## Task 5 — Add a baseline-preserving residual Teacher

**Problem addressed:** The current Teacher can destroy the Morgan baseline because all predictions pass through the multimodal bottleneck.

**Files:**

- `src/models/multimodal_teacher_student.py`
- `src/models/factory.py`
- `run_precomputed_experiment.py`
- `configs/model_multimodal_teacher.yaml`
- `tests/models/test_multimodal_teacher_student.py`
- `tests/test_precomputed_runner.py`

**Implementation:**

1. Add a Morgan-only symmetric dual-head baseline inside `MultiModalTeacher`, using shared drug encoders and `[A+B, |A-B|, A*B]` pair features.
2. Produce auxiliary organ/specific correction logits from the balanced modality summaries.
3. Define final logits exactly as:

   ```text
   final = morgan_baseline + sum(sigmoid(modality_gate) * modality_correction)
   ```

4. Train in two stages:
   - Stage A: train the Morgan baseline alone and select its checkpoint on validation macro AUPRC.
   - Stage B: load the selected baseline, freeze it, and train auxiliary projections, resamplers, gates, and correction heads.
5. During Stage B, retain the Stage-A checkpoint as an explicit fallback. Promote the fused Teacher only when its validation macro AUPRC exceeds the Stage-A baseline by the configured minimum promotion delta. Use `0.002` for the first experiment and treat it as a validation-only hyperparameter thereafter.
6. The Teacher cache encoder must still encode all enabled modalities for Student training. Cache provenance must identify the promoted Teacher checkpoint.
7. Persist separate validation metrics for baseline logits, each single-modality correction, combined Teacher logits, and the selected output.

**Tests:**

- With auxiliary gates forced to zero, Teacher logits must exactly equal Morgan baseline logits.
- At initialization, logits must be numerically close to the baseline because gates start near zero.
- A deliberately harmful auxiliary fixture must cause validation fallback to select the baseline.
- A useful auxiliary fixture must cause selection of the fused checkpoint.
- Both paths must remain pair-symmetric and hierarchy-compatible.

**Acceptance:**

- The promoted Teacher is never below the matched Morgan model on validation macro AUPRC by construction.
- Test performance remains untouched until baseline/fused selection is frozen.
- The final report clearly says whether baseline fallback or fused output was selected.

## Task 6 — Make distillation conditional on a qualified Teacher

**Problem addressed:** Distilling a weak or unproven Teacher transfers noise and cannot establish an architectural gain.

**Files:**

- `run_precomputed_experiment.py`
- `src/models/multimodal_teacher_student.py`
- `build_teacher_cache.py`
- `tests/test_precomputed_runner.py`
- `tests/features/test_cached_token_builder.py`

**Implementation:**

1. Refuse cache generation and Student training unless the Teacher artifact records successful validation promotion or an explicit baseline-fallback selection.
2. Tune `distillation_weight` and `distillation_temperature` using validation only. Include supervised-only Student (`distillation_weight = 0`) as a required control.
3. Keep the Teacher frozen and verify exact record alignment between Student and Teacher loaders using `pair_id`, not only positional `zip` ordering.
4. Persist per-seed Teacher–Student paired differences for macro/micro AUPRC, AUROC, Brier, and ECE.
5. Select Student checkpoints by validation macro AUPRC. Do not select on similarity to Teacher logits.

**Tests:**

- Shuffled Teacher records must fail on pair-ID mismatch.
- `distillation_weight = 0` must reduce exactly to supervised training.
- Cache hash or Teacher checkpoint mismatch must fail before optimizer creation.
- Teacher parameters must receive no gradients during Student training.

**Acceptance:**

- Student must preserve the promoted Teacher’s validation macro AUPRC within `0.005` absolute or improve it.
- If it fails this criterion, deploy the Teacher or redesign the Student; do not call distillation successful.

## Task 7 — Run gated ablations before the full sweep

**Problem addressed:** Without ablations, additional modalities can hide regressions and the source of any gain is unknowable.

**Execution order:**

1. Run all Task-0 configurations on `warm_pair / seed 42` using repaired producers and identical data contracts.
2. Rank by validation macro AUPRC. Open the test set only after each configuration is frozen.
3. Remove any modality that fails to improve validation macro AUPRC or consistently closes its gate.
4. Repeat the surviving Morgan-only, promoted Teacher, and Student models on all five warm-pair seeds.
5. Report paired mean, sample standard deviation, and bootstrap 95% confidence intervals.
6. Proceed to all five `cold_1` seeds only after warm-pair reproducibility is established.
7. Proceed to `cold_2` only after the corrected cold-export tests and `cold_1` run provenance pass.

**Promotion criteria:**

- Primary: paired improvement in macro AUPRC over Morgan-only.
- The advanced model is considered better only if the five-seed paired bootstrap 95% CI for `advanced - Morgan` is above zero.
- Also report micro AUPRC, macro/micro AUROC, Brier, ECE, common/medium/rare-label AP, and HGT coverage.
- No “accuracy percentage” language is permitted.

## Task 8 — Measure deployment cost after model selection

**Problem addressed:** CPU suitability is currently asserted but not measured.

**Files:**

- Add `scripts/benchmark_pair_inference.py`
- Add `tests/test_pair_inference_benchmark.py`
- Document measured results only after running on the target CPU.

**Implementation:**

1. Benchmark one-pair Student inference after 20 warm-up calls and at least 1,000 measured calls.
2. Report median, p95, and p99 latency, peak resident memory, checkpoint size, cached-feature size per drug, and thread count.
3. Verify prediction parity between the training checkpoint and the loaded CPU checkpoint on a fixed fixture.
4. Record CPU model, OS, PyTorch version, dtype, and commit hash.

**Acceptance:**

- No CPU-performance claim appears without a reproducible benchmark artifact.
- Prediction parity passes before latency numbers are accepted.

## Required verification commands

Run focused tests after each task, then the complete suite:

```bash
.venv/bin/python -m pytest -q tests/models/test_primekg_hgt.py tests/test_primekg_hgt_training.py
.venv/bin/python -m pytest -q tests/features/test_molformer_token_producer.py tests/models/test_molecular_mpnn.py
.venv/bin/python -m pytest -q tests/models/test_multimodal_teacher_student.py tests/test_precomputed_runner.py
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q src scripts run_precomputed_experiment.py
git diff --check
```

If the environment lives on the external disk, substitute its exact Python executable; do not create a second project environment merely to satisfy these example paths.

## Final deliverables

- Repaired and diagnostically validated HGT checkpoint/artifact producer.
- Leakage-safe warm/cold HGT token artifacts with coverage metadata.
- Lossless MolFormer and non-truncating MPNN artifacts.
- Balanced modality resampling and gates.
- Baseline-preserving residual Teacher with validation fallback.
- Pair-ID-verified Student distillation.
- Matched ablation table on v2 seed 42.
- Five-seed paired warm-pair comparison with confidence intervals.
- Cold-split evaluation only after inductive export verification.
- Reproducible CPU inference benchmark for the selected Student.


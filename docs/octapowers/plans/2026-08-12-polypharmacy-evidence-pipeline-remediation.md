# Polypharmacy Evidence Pipeline Remediation Plan

**Date:** 2026-08-12  
**Status:** Proposed; implementation must not begin until approved  
**Supersedes:** `2026-08-12-metric-improvement-roadmap.md` for execution details  
**Repository:** `/Users/atasesli/Desktop/VsCode/ai`

## 1. Objective

Build a reproducible evaluation and training pipeline that can legitimately answer all of the following:

1. How well does each model rank side effects for unseen pairs of known drugs?
2. How well does it generalize when one drug is unseen during training?
3. How well does it generalize when both drugs are unseen during training?
4. Are reported improvements stable across five seeds?
5. Are probability outputs calibrated on held-out validation data?
6. Which model components produce measurable improvements under identical splits?
7. Can every PDF number be traced to predictions, a checkpoint, a configuration, a dataset hash, and a benchmark manifest?

The final deliverable is not merely a higher score. It is a trustworthy evidence pipeline plus a report generated exclusively from that evidence.

## 2. Current-state findings that this plan must correct

The implementation starts from these verified facts:

- `train_unified_master_model.py` still uses a hard-coded absolute path, fixed 25 epochs, BCE, fixed thresholds, no validation selection, and only the old pair test split.
- `configs/benchmark.yaml` declares warm/Cold-1/Cold-2 and PU correction, but no code consumes those options.
- `src/data/benchmark_builder.py` does not build splits, does not derive labels from training triples, does not call `canonicalize_pair`, and stores only a label preview.
- `src/evaluation/reporting.py` ignores `runs_dir` and writes hard-coded scores and confidence intervals.
- `generate_advanced_pdf_report.py` contains a second, different hard-coded score set.
- The existing ChemBERTa attention block attends over sequence length one.
- No run predictions, immutable split artifacts, calibration data, bootstrap samples, or multi-seed results exist in the repository.

No task may preserve these behaviors behind a new wrapper.

## 3. Non-negotiable rules

1. **No metric literals in reports.** PDF, Markdown, README, and charts must load values from one validated summary JSON.
2. **No test-driven decisions.** Test labels and predictions may only be read after the checkpoint, thresholds, calibration, and model configuration are frozen.
3. **No cold-start claim without drug-disjoint assertions.** Pair-disjoint is named `warm_pair`; it is never called cold.
4. **No unobserved pair is called a confirmed negative.** Use `sampled_unlabeled` throughout code, artifacts, and reports.
5. **No `state-of-the-art`, clinical reliability, or actionable-clinical claim.** Those require external comparison or clinical validation outside this plan.
6. **No architecture claim without an ablation.** ChemBERTa, token attention, SIDER, PrimeKG, and hierarchy loss must each have a matched ablation.
7. **No hard-coded user-home paths.** Every executable path comes from CLI arguments or repository-relative defaults.
8. **Every reported aggregate uses the five fixed seeds:** `42`, `101`, `2024`, `31415`, `27182`.

## 4. Final repository layout

Create these files; retain existing model files unless a task explicitly replaces their use:

```text
build_benchmark.py
run_experiment.py
aggregate_results.py
configs/
  benchmark.yaml
  experiments/
    prevalence.yaml
    logistic.yaml
    symmetric_mlp.yaml
    unified.yaml
    unified_hierarchy.yaml
    unified_sider.yaml
    chemberta_mean.yaml
    chemberta_token_attention.yaml
    primekg_unified.yaml
    full_model.yaml
src/
  data/
    benchmark_builder.py
    schemas.py
    splitters.py
    negative_sampling.py
    validators.py
  evaluation/
    metrics.py
    thresholds.py
    calibration.py
    bootstrap.py
    reporting.py
  training/
    engine.py
    losses.py
    reproducibility.py
  features/
    chemberta_tokens.py
    primekg_embeddings.py
  models/
    symmetric_mlp.py
    token_cross_attention.py
tests/
  data/
    test_schemas.py
    test_splitters.py
    test_negative_sampling.py
    test_benchmark_builder.py
    test_validators.py
  evaluation/
    test_metrics.py
    test_thresholds.py
    test_calibration.py
    test_bootstrap.py
    test_reporting.py
  training/
    test_engine.py
    test_losses.py
    test_reproducibility.py
  models/
    test_pair_symmetry.py
    test_token_cross_attention.py
output/
  pdf/
artifacts/
  benchmarks/<benchmark_id>/
  runs/<run_id>/
results/
  benchmark_summary.json
  benchmark_summary.csv
```

## 5. Exact configuration contract

Replace `configs/benchmark.yaml` with this schema and reject unknown keys:

```yaml
schema_version: 1
benchmark_name: polypharmacy_v1
seeds: [42, 101, 2024, 31415, 27182]
top_k_labels: 100

paths:
  drugs_master: data/raw/drugs_master.csv
  side_effects: data/raw/side_effects.csv
  biosnap_ddi: data/external/ChChSe-Decagon_polypharmacy.csv
  sider: data/external/meddra_all_se.tsv
  primekg: data/external/primekg_kg.csv

partitions:
  warm_pair:
    train_fraction: 0.70
    validation_fraction: 0.15
    test_fraction: 0.15
  cold_1:
    train_drug_fraction: 0.70
    validation_new_drug_fraction: 0.10
    test_new_drug_fraction: 0.20
  cold_2:
    train_drug_fraction: 0.70
    validation_new_drug_fraction: 0.10
    test_new_drug_fraction: 0.20

sampling:
  controls_per_positive_pair: 1
  primekg_degree_bins: 10
  molecular_similarity_bins: 5
  maximum_attempts_per_control: 10000
  pu_prior_multipliers: [1.0, 2.0, 4.0]

bootstrap:
  resamples: 2000
  confidence_level: 0.95
  seed: 8675309
```

Validation requirements:

- partition fractions must sum to `1.0` within `1e-9`;
- seeds must be unique integers;
- `top_k_labels`, bin counts, attempts, and resamples must be positive integers;
- every configured input path must exist before benchmark construction;
- every input file must contain the required columns defined in `src/data/schemas.py`.

## 6. Exact artifact schemas

### 6.1 Canonical observed-positive triple table

`observed_positive_triples.parquet`:

| Column | Type | Rule |
|---|---|---|
| `pair_id` | string | SHA-256 of `drug_a + "\0" + drug_b`, first 24 hex chars |
| `drug_a` | string | lexicographically smaller canonical DrugBank ID |
| `drug_b` | string | lexicographically larger canonical DrugBank ID |
| `label_cui` | string | modeled UMLS CUI |
| `observation_status` | string | always `observed_positive` |

Reject self-pairs. Deduplicate on `(drug_a, drug_b, label_cui)`.

### 6.2 Pair split table

Each scenario/seed writes `pairs.parquet`:

| Column | Type | Allowed values |
|---|---|---|
| `pair_id` | string | canonical pair hash |
| `drug_a` | string | DrugBank ID |
| `drug_b` | string | DrugBank ID |
| `split` | categorical | `train`, `validation`, `test` |
| `scenario` | categorical | `warm_pair`, `cold_1`, `cold_2` |
| `observation_status` | categorical | `observed_positive`, `sampled_unlabeled` |
| `source_positive_pair_id` | nullable string | populated only for corrupted sampled controls |

### 6.3 Label matrix metadata

`labels.json` must contain the full ordered list, never a preview:

```json
{
  "selection_split": "train",
  "top_k": 100,
  "labels": [
    {"index": 0, "cui": "...", "name": "...", "train_positive_count": 123}
  ],
  "out_of_vocabulary_positive_counts": {
    "train": 0,
    "validation": 0,
    "test": 0
  }
}
```

### 6.4 Benchmark manifest

`manifest.json` must include:

- schema version, benchmark ID, scenario, seed, creation timestamp;
- Git SHA;
- every source path, byte size, SHA-256, and declared dataset version;
- row counts before and after every mapping/filter/deduplication step;
- full split pair counts and positive/unlabeled counts;
- unique drug counts by split;
- train, validation-new, and test-new drug hashes;
- label-file SHA-256;
- split-table SHA-256;
- sampler diagnostics;
- all validator results with `passed: true`;
- canonical JSON SHA-256 calculated before adding `manifest_hash`.

### 6.5 Run directory

Every run writes:

```text
artifacts/runs/<run_id>/
  config.resolved.json
  environment.json
  training_history.csv
  checkpoint_best.pt
  validation_logits.parquet
  validation_predictions.parquet
  test_logits.parquet
  test_predictions.parquet
  thresholds.json
  calibration.json
  metrics.json
  per_label_metrics.csv
  completion.json
```

`run_id` is:

```text
<model_name>__<scenario>__seed_<seed>__<benchmark_hash_12>
```

Write `completion.json` last and only after all files are flushed and hashes verified. Aggregation ignores any run without `completion.json` containing `"status": "complete"`.

## 7. Implementation phases

### Phase 0 — Restore portability and create enforceable tests

**Modify:** all executable `train_*.py`, `src/data/benchmark_builder.py`, `src/evaluation/reporting.py`, `generate_advanced_pdf_report.py`.  
**Create:** `src/training/reproducibility.py`, initial tests.

Steps:

1. Write a repository-wide test that scans non-test Python source for `/Users/`, `.gemini/`, and `Downloads/`; allow no matches.
2. Add `repository_root()` using `Path(__file__).resolve()` from the relevant entry point. CLI arguments override defaults.
3. Add `set_global_seed(seed)` that seeds `random`, NumPy, PyTorch CPU, CUDA, and MPS where supported; enable deterministic algorithms where available and record warnings otherwise.
4. Add `environment.json` collection for Python, platform, PyTorch, NumPy, pandas, scikit-learn, RDKit, transformers, device, Git SHA, and dirty-worktree status.
5. Make old training scripts fail with a clear message directing callers to `run_experiment.py`, or refactor them to call the new engine. They must not retain an independent evaluation path.
6. Add `pytest>=8,<9` and `pytest-cov>=5,<6` to `requirements-dev.txt`.

Acceptance gate:

- source scan finds zero hard-coded home paths;
- `python -m compileall -q .` passes;
- import tests pass;
- no report or summary file contains numeric model results in source code.

### Phase 1 — Build canonical data before selecting labels

**Modify:** `src/data_loader.py`.  
**Create:** `src/data/schemas.py`, rewrite `src/data/benchmark_builder.py`, tests.

Steps:

1. Validate source columns before any processing.
2. Map STITCH/PubChem identifiers to DrugBank IDs; record every unmapped reason separately.
3. Canonicalize pairs before filtering and grouping.
4. Remove self-pairs and exact duplicate triples.
5. Persist the complete canonical observed-positive triple table before label selection.
6. Unit-test identifier parsing with representative `CID`, `CID0`, `CID1`, missing, malformed, and already-DrugBank values.

Acceptance gate:

- reversed input pairs produce identical `pair_id`;
- duplicate source rows produce one canonical triple;
- all discarded rows have a counted reason;
- re-running with identical inputs produces byte-identical Parquet content after metadata normalization.

### Phase 2 — Implement deterministic warm and cold splitters

**Create:** `src/data/splitters.py`, `src/data/validators.py`, tests.

Use NumPy `Generator(PCG64(seed))`; do not use global random state.

#### Warm-pair algorithm

1. Work on unique positive pair IDs without reading label identities.
2. Shuffle pair IDs with the seeded generator.
3. Assign the first 70% to train, next 15% to validation, last 15% to test.
4. For every drug absent from train but present in validation/test, move the lexicographically smallest pair containing that drug from validation/test to train.
5. Rebalance by moving eligible train pairs into the undersized split only when both endpoints remain represented in train after the move.
6. Stop when target counts are within one pair or no legal move remains; record achieved fractions.

Required assertion: every validation/test drug occurs in train.

#### Cold drug partition

1. Compute per-drug positive-pair degree only for stratifying the drug partition; do not expose this value as a model feature.
2. Bin drugs into degree deciles with deterministic tie handling `(degree, drug_id)`.
3. Within each bin, seeded-shuffle and allocate 70% `train_drugs`, 10% `validation_new_drugs`, 20% `test_new_drugs`.
4. Ensure drug sets are pairwise disjoint.

#### Cold-1 pair selection

- train: both endpoints in `train_drugs`;
- validation: exactly one endpoint in `validation_new_drugs`, the other in `train_drugs`;
- test: exactly one endpoint in `test_new_drugs`, the other in `train_drugs`;
- exclude all other pair combinations.

#### Cold-2 pair selection

- train: both endpoints in `train_drugs`;
- validation: both endpoints in `validation_new_drugs`;
- test: both endpoints in `test_new_drugs`;
- exclude all other pair combinations.

Fail benchmark construction if any required split has fewer than 100 positive pairs or fewer than 10 positive instances for more than 50% of the selected labels. Do not silently fall back to a warm split.

Acceptance gate:

- no pair or reversed pair overlaps splits;
- Cold-1 test has exactly one test-new endpoint per pair;
- Cold-2 test has exactly two test-new endpoints per pair;
- no test-new drug occurs in train or validation;
- five seeds produce five deterministic but not necessarily identical partitions.

### Phase 3 — Select labels from training triples only

**Modify:** benchmark builder and validators.  
**Create:** `tests/data/test_benchmark_builder.py`.

For each scenario and seed independently:

1. Join canonical triples to train pair IDs only.
2. Count unique train pairs per CUI.
3. Sort labels by descending count, then ascending CUI for deterministic ties.
4. Select first 100.
5. Freeze label order and apply it unchanged to validation/test.
6. Persist full label metadata and OOV counts.

Mandatory mutation test: alter validation and test labels/frequencies and prove `labels.json` remains byte-identical.

Acceptance gate: the mutation test passes and the manifest hashes the full label file.

### Phase 4 — Implement degree-matched sampled-unlabeled controls

**Create:** `src/data/negative_sampling.py`, `src/training/losses.py`, tests.

Candidate construction:

1. Build `known_positive_pairs` from all canonical observed-positive pairs across every CUI.
2. Build per-drug matching attributes from non-DDI inputs only:
   - PrimeKG biological-edge degree after excluding all drug-drug interaction/outcome relations;
   - Morgan fingerprint similarity to the retained endpoint.
3. For each observed-positive pair, randomly choose one endpoint to corrupt.
4. Draw a replacement drug from the same scenario/split-allowed drug pool.
5. Require replacement PrimeKG degree decile to match the corrupted drug's decile.
6. Require pair Morgan-similarity bin to match the source pair's bin.
7. Reject self-pairs, known positives, duplicates, cross-split violations, and scenario violations.
8. After 10,000 unsuccessful attempts, relax molecular bin by one adjacent bin; never relax degree bin or scenario constraints. Record every relaxation.
9. Generate exactly one control per positive pair. Fail rather than under-fill.

PU loss specification for each label `l`:

```text
positive_loss = softplus(-positive_logits)
negative_loss = softplus(unlabeled_logits)
positive_as_negative_loss = softplus(positive_logits)
pi_l = clip(observed_train_prevalence_l * prior_multiplier, 1e-6, 0.5)
positive_risk = pi_l * mean(positive_loss)
negative_risk = mean(negative_loss) - pi_l * mean(positive_as_negative_loss)
nnpu_risk = positive_risk + max(0, negative_risk)
```

Select `prior_multiplier` from `[1.0, 2.0, 4.0]` using validation macro AP only. Labels without a positive in the current batch contribute no positive-risk term; use epoch-level balanced sampling so every train-positive label appears regularly.

Acceptance gate:

- zero sampled controls overlap known positives;
- control count equals positive-pair count;
- scenario invariants hold after corruption;
- degree-decile match rate is 100%;
- relaxed molecular-bin rate is reported;
- tests cover empty bins, attempt exhaustion, duplicate rejection, and deterministic output.

### Phase 5 — Make metrics authoritative

**Rewrite:** `src/evaluation/metrics.py`.  
**Create:** thresholds, calibration, bootstrap modules and tests.

Required metrics:

- macro AP over labels containing both classes;
- micro AP over all pair-label instances;
- macro and micro AUROC over valid labels;
- pair-level Precision@5, Recall@5, NDCG@5;
- per-label AP lift = `AP / train_prevalence`;
- Brier score;
- ECE with 15 equal-width probability bins;
- sensitivity, specificity, PPV, NPV, and F1 at frozen thresholds;
- hierarchy violation rate = fraction where `p_specific > p_parent_organ`.

Metric output must include:

- value;
- numerator/denominator where applicable;
- number and IDs of excluded undefined labels;
- train prevalence used for lift;
- common/medium/rare frequency stratum.

Frequency strata are derived from train counts only:

- rare: bottom 33% of modeled labels;
- medium: middle 34%;
- common: top 33%;
- ties resolved by CUI ordering.

Threshold selection:

1. Candidate thresholds are unique validation probabilities plus `0` and `1`.
2. For each label, choose the threshold with maximum recall subject to `PPV >= configured_min_ppv`.
3. If no candidate meets PPV, choose maximum F1.
4. If validation has fewer than 10 positives for a label, use the global threshold selected by micro F1.
5. Persist threshold, selection rule, positive count, PPV, recall, and F1.

Calibration:

- fit one scalar temperature for organ logits and one for specific logits using validation NLL;
- constrain temperature to `[0.05, 10.0]`;
- use LBFGS with maximum 100 iterations;
- fit before threshold selection;
- retain calibration only if validation NLL improves and validation macro AP changes by less than `1e-10`.

Bootstrap:

- resample pair rows with replacement 2,000 times;
- calculate each metric on each resample;
- use percentile CI at 2.5% and 97.5%;
- use bootstrap seed `8675309`;
- record failed/undefined resamples.

Acceptance gate:

- exact-value unit tests use hand-computed fixtures;
- metric parity tests use scikit-learn where applicable;
- test-label mutation cannot alter thresholds or temperatures;
- identical predictions produce identical bootstrap output.

### Phase 6 — Build one validation-driven training engine

**Create:** `src/training/engine.py`, `run_experiment.py`.  
**Refactor:** old training scripts into wrappers or remove their independent logic.

Training protocol:

1. Load one immutable benchmark manifest and verify all hashes.
2. Load model configuration and write `config.resolved.json` before training.
3. Train for at most 100 epochs.
4. Evaluate validation macro AP after every epoch.
5. Save a checkpoint only when validation macro AP improves by at least `1e-4`.
6. Stop after five consecutive non-improving epochs.
7. Reload the best checkpoint.
8. Produce validation logits.
9. Fit temperature scaling on validation logits.
10. Produce calibrated validation probabilities.
11. Select thresholds from validation only.
12. Freeze checkpoint hash, temperatures, and thresholds.
13. Produce test logits exactly once.
14. Apply frozen calibration and thresholds.
15. Write test predictions and metrics.
16. Write `completion.json` last.

The test loader object must not be constructed until step 13. Enforce this with a trainer state enum:

```text
INITIALIZED -> TRAINING -> MODEL_SELECTED -> VALIDATION_FROZEN -> TEST_EVALUATED -> COMPLETE
```

Any attempt to access test data before `VALIDATION_FROZEN` raises `RuntimeError`.

Acceptance gate:

- synthetic overfit test proves best-epoch restoration;
- patience and minimum-delta tests pass;
- test-access guard test passes;
- changing test labels does not alter the selected checkpoint hash, temperatures, or thresholds;
- rerunning the same seed/config/benchmark produces identical predictions within `1e-7` on CPU.

### Phase 7 — Establish baselines before architecture changes

Implement these experiment configs against identical benchmark hashes:

1. `prevalence`: predicts train label prevalence for every pair.
2. `logistic`: one-vs-rest logistic regression on symmetric molecular features.
3. `symmetric_mlp`: shared drug encoder plus `[sum, abs-diff, product]`; no hierarchy, SIDER, ChemBERTa, or PrimeKG.
4. `unified`: current tensor/MLP decoder, migrated to the new trainer without changing its model math.

Run all four across three scenarios and five seeds before Phase 8.

Promotion gate:

- centralized metrics and run artifacts exist for 60 runs (`4 models × 3 scenarios × 5 seeds`);
- all run directories are complete;
- aggregation can be reproduced from predictions alone;
- report generation remains disabled until aggregation validation passes.

### Phase 8 — Implement controlled model additions

Each addition is implemented and evaluated separately. Do not combine components before their individual ablations finish.

#### 8A. Hierarchy consistency

1. Replace keyword-only SOC mapping with a versioned CUI-to-SOC mapping file where available.
2. Preserve keyword mapping only as `mapping_source: keyword_fallback`.
3. Add loss:

   `hierarchy_loss = mean(relu(sigmoid(specific_logits) - sigmoid(parent_organ_logits)))`

4. Tune weight over `[0.0, 0.05, 0.1, 0.25, 0.5]` on validation macro AP, breaking ties with lower violation rate.

#### 8B. SIDER prior

1. Build priors using only versioned SIDER input, independent of BioSNAP target labels.
2. Convert binary prior to smoothed log-odds with epsilon `1e-4`.
3. Tune fusion coefficient on validation.
4. Report mapped-drug and mapped-label coverage.

#### 8C. Mean-pooled ChemBERTa

1. Pin exact model and tokenizer revisions.
2. Cache embeddings with attention-mask-aware mean pooling.
3. Store embedding manifest and drug ordering.
4. Use missing-SMILES indicator plus a train-derived mean vector; do not silently impute without the indicator.

Each component must run across all scenarios and seeds. Promote only if mean validation macro AP improves over `unified` and at least three of five seeds improve in Cold-1 or Cold-2.

### Phase 9 — Implement real token-level cross-attention

**Create:** `src/features/chemberta_tokens.py`, `src/models/token_cross_attention.py`, tests.

Architecture contract:

1. Cache ChemBERTa `last_hidden_state` and attention mask per drug, maximum length 256.
2. Project tokens from ChemBERTa hidden size to 128 dimensions.
3. Apply bidirectional `nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)` with the other molecule as key/value.
4. Pass `key_padding_mask = ~attention_mask.bool()`.
5. Residual-add and LayerNorm each cross-conditioned sequence.
6. Masked-mean-pool each sequence.
7. Construct symmetric pair representation `[a+b, abs(a-b), a*b]`.
8. Feed the same organ and specific heads used by the matched mean-pooled ablation.

Required tests:

- ordinary SMILES produce token length greater than one;
- padding changes neither pooled embeddings nor predictions;
- swapping drugs changes every output probability by at most `1e-6` in eval mode;
- attention weights for padding positions are zero within `1e-7`;
- gradient reaches both token projectors and both attention directions.

Promotion gate: token attention must improve five-seed mean validation macro AP over mean-pooled ChemBERTa and improve at least one cold scenario without degrading the other by more than its 95% CI width.

### Phase 10 — Add leakage-safe PrimeKG embeddings

**Create:** `src/features/primekg_embeddings.py`, provenance tests.

1. Enumerate allowed PrimeKG relation names explicitly in configuration.
2. Reject every drug-drug edge and any relation derived from adverse-event/DDI outcomes.
3. Map DrugBank IDs through stable identifiers first; normalized names are fallback and are recorded.
4. Train a relation-aware encoder using allowed biological edges only.
5. Produce an embedding for an unseen drug using its permitted attributes and neighbors without any DDI target edges.
6. Fit all graph normalization on training drugs and freeze it.
7. Add `primekg_degree_only` and `primekg_embedding` ablations.

Required provenance test: inject a sentinel DDI outcome edge and prove it is rejected before graph construction.

Promotion gate: PrimeKG embeddings must outperform both no-PrimeKG and degree-only ablations on at least one cold scenario with no statistically meaningful regression on the other.

### Phase 11 — Aggregate only genuine completed runs

**Rewrite:** `src/evaluation/reporting.py`.  
**Create:** `aggregate_results.py`, reporting tests.

Algorithm:

1. Discover `completion.json` files under `artifacts/runs`.
2. Verify every run hash and benchmark hash.
3. Group by `(model_name, scenario)`.
4. Require exactly the five configured seeds; fail on missing or duplicate seeds.
5. Recompute metrics from each run's test predictions rather than trusting `metrics.json`.
6. Assert recomputed values match stored metrics within `1e-10`.
7. Calculate mean and sample standard deviation across seeds.
8. Calculate the pair-level bootstrap CI from concatenated seed-tagged predictions, resampling within seed.
9. Select the champion by:
   - highest mean Cold-1 macro AP plus Cold-2 macro AP;
   - tie within `1e-4`: higher warm macro AP;
   - further tie: lower parameter count.
10. Write `benchmark_summary.json` and flattened CSV.

The summary must include every model/scenario, not only the champion.

Acceptance gate:

- source contains no score literals;
- deleting one seed causes aggregation to fail;
- altering stored `metrics.json` causes a parity failure;
- rebuilding from unchanged runs produces byte-identical summary JSON except an optional excluded timestamp field.

### Phase 12 — Generate the report from the summary

**Rewrite:** `generate_advanced_pdf_report.py`.  
**Update:** README, Markdown guide, other PDF generators.

CLI:

```bash
python generate_advanced_pdf_report.py \
  --summary results/benchmark_summary.json \
  --output output/pdf/Polypharmacy_AI_Comprehensive_Report.pdf
```

Required report sections:

1. Research objective and non-clinical status.
2. Data sources with versions, mapped counts, and hashes.
3. Exact warm/Cold-1/Cold-2 definitions.
4. Positive and sampled-unlabeled construction.
5. Model architecture matching the selected run configuration.
6. Results table for all three scenarios with mean, SD, and 95% CI.
7. Primary metrics: macro/micro AP, Precision@5, Recall@5, NDCG@5.
8. Secondary discrimination and calibration metrics.
9. Common/medium/rare label performance.
10. Ablation table.
11. Limitations: retrospective data, sampled-unlabeled controls, no prospective validation, no causal interpretation.
12. Reproducibility appendix: Git SHA, benchmark hash, run IDs, seeds, package versions.

Forbidden report text unless separately supported:

- `state-of-the-art`;
- `highly actionable for clinicians`;
- `clinical reliability`;
- `trained on 2.42 million ChEMBL molecules` when ChEMBL was only counted/indexed;
- `cold split` without naming Cold-1 or Cold-2;
- `GNN` when the selected model performs no graph message passing;
- atom-ring interpretation without validated atom-level attribution.

Visual acceptance:

- no section heading orphaned on the prior page;
- no page with more than 45% unused vertical area unless intentionally a cover;
- no diagram labels overlap arrows;
- charts do not mix percentage and unit-scale metrics without explicit normalization labels;
- rendered PDF pages show no clipped footer, table, glyph, or image;
- text extraction contains every metric and run identifier shown visually.

## 8. Required command sequence

After dependencies and data are available, the teammate must run exactly this sequence:

```bash
python -m compileall -q .
python -m pytest -q
python build_benchmark.py --config configs/benchmark.yaml
python -m pytest -q tests/data

python run_experiment.py --experiment configs/experiments/prevalence.yaml --all-benchmarks
python run_experiment.py --experiment configs/experiments/logistic.yaml --all-benchmarks
python run_experiment.py --experiment configs/experiments/symmetric_mlp.yaml --all-benchmarks
python run_experiment.py --experiment configs/experiments/unified.yaml --all-benchmarks

python aggregate_results.py --runs artifacts/runs --output results/benchmark_summary.json
python -m pytest -q tests/evaluation tests/training

python generate_advanced_pdf_report.py \
  --summary results/benchmark_summary.json \
  --output output/pdf/Polypharmacy_AI_Comprehensive_Report.pdf

pdftotext -layout output/pdf/Polypharmacy_AI_Comprehensive_Report.pdf -
pdftoppm -png output/pdf/Polypharmacy_AI_Comprehensive_Report.pdf /tmp/polypharmacy-report-page
```

Architecture experiments are added to the `run_experiment.py` sequence only after the baseline promotion gate passes.

## 9. Completion checklist

The work is complete only when all boxes are true:

- [ ] No executable source contains a hard-coded user-home path.
- [ ] No report or summary source contains model score literals.
- [ ] Warm, Cold-1, and Cold-2 split validators pass for five seeds.
- [ ] Test-new drugs never occur in train or validation.
- [ ] Label vocabulary is derived exclusively from training triples.
- [ ] Sampled controls never overlap known positive pairs.
- [ ] PU loss and matching diagnostics are recorded.
- [ ] Validation selects checkpoints, calibration, thresholds, and tunable weights.
- [ ] Test data access is blocked before validation freeze.
- [ ] All required metrics and uncertainty estimates are recomputable from predictions.
- [ ] Every promoted component has a matched ablation.
- [ ] The champion selection rule executes automatically.
- [ ] Five completed seeds exist for every reported model/scenario group.
- [ ] The PDF reads only `benchmark_summary.json` and verified run metadata.
- [ ] PDF pages pass rendered visual inspection.
- [ ] README, Markdown, JSON, CSV, and PDF values agree exactly.
- [ ] Clinical and state-of-the-art claims are absent.

## 10. Rollback and compatibility

- Do not delete old report PDFs during implementation; move superseded outputs to `output/pdf/archive/` after the new report passes verification.
- Do not overwrite existing checkpoints. New checkpoints live only under run IDs.
- Preserve current model classes for reproducibility, but route evaluation through the new engine.
- If a new benchmark invalidates old metrics, retain them only under `legacy_unverified` in documentation; never mix them with verified results.
- Any failure in manifest, hash, split, seed, or metric parity validation aborts aggregation and report generation with a non-zero exit code.


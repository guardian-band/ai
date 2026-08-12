# Polypharmacy Model Metric-Improvement Implementation Plan

**Date:** 2026-08-12  
**Status:** Proposed for teammate handoff  
**Primary objective:** Produce trustworthy, reproducible improvements in polypharmacy side-effect ranking, especially macro/micro average precision and top-k recall, under warm-start and true cold-start evaluation.

## 1. Scope and boundaries

This plan covers:

- leakage-resistant warm, Cold-1, Cold-2, and optional scaffold-disjoint benchmarks;
- train-only label selection and reproducible multi-seed splits;
- positive-unlabeled-aware, degree-matched negative sampling;
- validation-driven checkpointing, threshold selection, and calibration;
- clinically meaningful ranking, discrimination, and calibration metrics;
- replacement of the current length-one attention block with token-level molecular cross-attention;
- explicit PrimeKG relation-aware embeddings and hierarchy-consistency training;
- machine-readable experiment outputs and generated reports.

This plan does **not** claim clinical efficacy, modify the existing PDFs by hand, introduce a serving API, or perform prospective/external clinical validation. Until external validation exists, all user-facing language must describe the system as a research model rather than a clinical decision-support product.

## 2. Chosen technical approach

The implementation will use these decisions consistently:

1. **Evaluation precedes optimization.** Freeze benchmark definitions and metric code before changing the model.
2. **Drug identity defines cold start.** Cold-1 contains one held-out drug; Cold-2 contains two held-out drugs. No held-out drug may occur in training or validation.
3. **Labels are selected from training data only.** Test-set frequency must never influence the top-100 label vocabulary.
4. **Unobserved interactions are unlabeled, not confirmed safe.** Training uses degree-matched, type-specific corrupted pairs with positive-unlabeled risk correction. Reports explicitly identify evaluation negatives as sampled unlabeled controls.
5. **The primary selection metric is validation macro average precision.** Test data is evaluated once per frozen run.
6. **Cross-attention operates on ChemBERTa token sequences.** Mean-pooled vectors remain available as an ablation, but the existing length-one attention block is removed.
7. **PrimeKG is represented by relation-aware inductive drug embeddings.** Only biological edges available independently of the held-out DDI labels may be used. DDI outcome edges must not enter feature construction.
8. **All headline metrics are multi-seed estimates.** Use seeds `42`, `101`, `2024`, `31415`, and `27182`, with 95% bootstrap confidence intervals over out-of-fold/test predictions.

## 3. Target repository structure

Create or extend the following paths:

```text
configs/
  benchmark.yaml
  model_unified.yaml
  model_token_cross_attention.yaml
src/
  data/
    benchmark_builder.py
    negative_sampling.py
    split_validation.py
  evaluation/
    metrics.py
    calibration.py
    bootstrap.py
    reporting.py
  training/
    engine.py
    losses.py
    thresholding.py
  features/
    chemberta_tokens.py
    primekg_embeddings.py
  models/
    token_cross_attention.py
tests/
  data/
  evaluation/
  training/
  models/
artifacts/
  benchmarks/<benchmark_id>/
  runs/<run_id>/
results/
  benchmark_summary.json
  benchmark_summary.csv
```

Keep the existing training scripts temporarily as compatibility entry points. They should delegate to shared modules rather than retain duplicate split, metric, and training logic.

## 4. Artifact contracts

### 4.1 Benchmark manifest

Each benchmark directory must contain `manifest.json` with:

- source file paths, versions, sizes, and SHA-256 hashes;
- canonical benchmark configuration and benchmark ID;
- label vocabulary and train-only frequencies;
- split strategy, seed, and pair/drug counts;
- positive and sampled-unlabeled counts per split and label;
- train/validation/test drug sets;
- assertions proving pair and drug separation;
- molecular scaffold overlap statistics when scaffold evaluation is enabled.

Split tables must include at least:

```text
drug_a, drug_b, split, scenario, label_cui, target, observation_status
```

`observation_status` is one of `observed_positive` or `sampled_unlabeled`; do not name sampled controls `confirmed_negative`.

### 4.2 Run result

Each run directory must contain:

- `config.json` with fully resolved parameters;
- `environment.json` with Python/package/device versions and Git SHA;
- `checkpoint_best.pt` selected on validation macro AP;
- `validation_predictions.parquet` and `test_predictions.parquet`;
- `thresholds.json` and `calibration.json` fitted on validation only;
- `metrics.json`, `per_label_metrics.csv`, and `training_history.csv`;
- split benchmark ID and all random seeds.

## 5. Ordered implementation tasks

### Task 1 — Make the project portable and testable

**Files:** all `train_*.py`, `src/features/*.py`, `src/data_ingestion/*.py`, `requirements.txt`, new `requirements-dev.txt`, new config files.

1. Add failing smoke tests that import every training, feature, data, model, and evaluation module.
2. Replace hard-coded `/Users/acelyayildiz/...` roots with CLI/config-provided paths resolved relative to the repository by default.
3. Move shared seed initialization into `src/training/engine.py`; seed Python, NumPy, and PyTorch, and record deterministic settings.
4. Add `pytest` to `requirements-dev.txt`; pin direct dependencies to reproducible compatible ranges and record the resolved environment in every run.
5. Add a `--dry-run` path that validates configuration and required input columns without training.

**Acceptance criteria:**

- `python -m compileall -q .` succeeds.
- `pytest -q` succeeds on a machine with project dependencies installed.
- No executable source file contains an absolute user-home path.
- Dry-run failures name the missing file or column and exit non-zero.

### Task 2 — Centralize canonical pair and label construction

**Files:** `src/data_loader.py`, new `src/data/benchmark_builder.py`, tests under `tests/data/`.

1. Canonicalize every pair as `(min(drug_a, drug_b), max(drug_a, drug_b))` before grouping or splitting.
2. Deduplicate `(drug_a, drug_b, CUI)` rows and persist mapping/drop statistics.
3. Split candidate pairs before deriving the label vocabulary.
4. Select top labels only from observed-positive training rows; persist the ordered vocabulary.
5. Apply that frozen vocabulary unchanged to validation and test, recording out-of-vocabulary positives rather than silently discarding them.
6. Add tests for reversed pairs, duplicated triples, missing drug mappings, missing CUIs, and train-only label selection.

**Acceptance criteria:**

- Reversing input drug order does not alter any pair ID or label.
- Modifying test-label frequencies cannot change the selected vocabulary.
- The manifest reports every filtered or unmapped row count.

### Task 3 — Implement valid warm and cold benchmarks

**Files:** new `src/data/split_validation.py`, `src/data/benchmark_builder.py`, `configs/benchmark.yaml`, corresponding tests.

Implement four scenarios:

- **Warm:** pairs are disjoint; every test drug appears in training.
- **Cold-1:** exactly one drug in each test pair belongs to the held-out drug set; held-out drugs occur nowhere in train/validation.
- **Cold-2:** both drugs belong to the held-out drug set; held-out drugs occur nowhere in train/validation.
- **Scaffold-disjoint (optional benchmark, enabled after the first three pass):** assign drugs to Bemis-Murcko scaffold groups and prevent scaffold-group overlap across train and test.

Generate separate validation sets with the same semantics as each test scenario. Stratify approximately by label cardinality and drug degree, without leaking test labels into training decisions.

Add hard assertions:

- no pair overlaps across splits;
- no reversed-pair overlap;
- Cold-1/Cold-2 held-out drugs never appear in train or validation;
- warm test drugs all appear in train;
- label vocabulary was fitted on train only.

**Acceptance criteria:**

- All split assertions execute whenever a benchmark is loaded, not only during creation.
- Five seeded manifests are generated for every required scenario.
- The report lists pair counts, unique-drug counts, label prevalence, and drug/scaffold overlap.

### Task 4 — Replace naive negative sampling

**Files:** new `src/data/negative_sampling.py`, new `src/training/losses.py`, benchmark builder and tests.

1. Build a sampler that corrupts one endpoint of an observed positive while preserving the split's allowed drug pool and canonical pair ordering.
2. Match sampled-unlabeled controls to positives by endpoint degree decile and, where feasible, molecular-similarity bin.
3. Reject every known positive for any modeled side-effect type; keep a hash set for constant-time checks.
4. Sample independently inside each split so no sampled control crosses split boundaries.
5. Estimate class prior from training data and implement a non-negative positive-unlabeled loss per side-effect label. Keep weighted BCE as an explicit ablation, not the default claim.
6. Keep validation/test prevalence fixed by configuration and report it. Do not force a hidden 1:1 prevalence.

**Acceptance criteria:**

- No sampled-unlabeled row is present in the known-positive set.
- Degree-distribution distance between positives and controls is reported and covered by a configured tolerance.
- Tests prove that cold held-out drugs cannot leak into the training sampler.
- Results label these examples as sampled unlabeled rather than true negatives.

### Task 5 — Build one authoritative metrics package

**Files:** new `src/evaluation/metrics.py`, `bootstrap.py`, `reporting.py`; remove duplicated metric helpers from training scripts after parity tests pass.

Implement:

**Primary metrics**

- macro and micro average precision using `average_precision_score`;
- Precision@5, Recall@5, and NDCG@5 per pair;
- per-label lift: `AP / positive_prevalence`.

**Secondary metrics**

- macro/micro AUROC where both classes are present;
- Brier score and expected calibration error;
- sensitivity, specificity, PPV, NPV, and F1 at validation-selected thresholds.

Rules:

- Do not compute trapezoidal PR AUC under the name average precision.
- Report undefined labels and exclude them transparently from macro aggregates.
- Report common, medium-frequency, and rare-label strata based on training frequency only.
- Bootstrap at the drug-pair level with 2,000 resamples and a fixed bootstrap seed.
- Produce metrics separately for warm, Cold-1, Cold-2, and scaffold scenarios.

**Acceptance criteria:**

- Unit tests compare every metric against small hand-computed fixtures or scikit-learn.
- Swapping label prevalence changes lift and calibration outputs as expected.
- `metrics.json` contains point estimates, 95% confidence intervals, denominators, and undefined-label counts.

### Task 6 — Use validation for all model choices

**Files:** new `src/training/engine.py`, `thresholding.py`, `calibration.py`; refactor all champion training scripts.

1. Evaluate validation macro AP after each epoch.
2. Save the best checkpoint with configurable patience, default `5`, and minimum improvement, default `1e-4`.
3. Fit per-label thresholds on validation predictions only. Default objective: maximize recall subject to a configured minimum PPV; labels without enough positives use a global validation threshold.
4. Fit temperature scaling on validation logits separately for organ and specific heads.
5. Tune learning rate, dropout, loss weights, hierarchy weight, and PU prior using validation metrics only.
6. Reload the frozen best checkpoint, calibration parameters, and thresholds before making test predictions.
7. Make the test evaluator read-only: it must not update checkpoints, thresholds, calibration, or configuration.

**Acceptance criteria:**

- A test that changes test labels cannot alter model selection, thresholds, or calibration.
- Training history identifies the selected epoch and validation objective.
- Calibration is reported before and after scaling; reject calibration changes that materially reduce the primary validation metric.

### Task 7 — Establish trustworthy baselines and ablations

**Files:** refactor `train_fast_baselines.py`, `train_unified_master_model.py`, `train_advanced_gated_model.py`; new configs.

Run every model on identical benchmark IDs:

1. prevalence-only and drug-frequency baselines;
2. logistic regression on symmetric molecular features;
3. symmetric pair MLP without graph, hierarchy, SIDER, or ChemBERTa;
4. unified tensor/MLP model;
5. hierarchy-only addition;
6. SIDER-prior addition;
7. ChemBERTa mean-pooled addition;
8. token cross-attention addition;
9. PrimeKG relation-aware embedding addition;
10. full model.

Use matched seeds and, where practical, matched training budgets. Report delta macro AP and confidence interval relative to the symmetric MLP.

**Acceptance criteria:**

- Every claimed component has an ablation on the same splits.
- The “champion” label is assigned automatically from the frozen selection metric, never hard-coded in a PDF.
- Complexity is retained only when its confidence interval and cold-start behavior justify it.

### Task 8 — Replace length-one attention with token cross-attention

**Files:** new `src/features/chemberta_tokens.py`, `src/models/token_cross_attention.py`; update `train_chemberta_cross_attention_model.py`; tests under `tests/models/`.

1. Cache the last hidden-state token sequence and attention mask for each valid SMILES using the existing ChemBERTa checkpoint and tokenizer version.
2. Persist model name, revision, tokenizer revision, maximum token length, dtype, and drug ordering in the feature manifest.
3. Project token embeddings to the model dimension, then compute bidirectional cross-attention over valid tokens with padding masks.
4. Pool cross-conditioned token sequences with masked attention pooling.
5. Construct symmetric pair features from pooled A/B outputs so swapping drugs changes predictions by no more than numerical tolerance.
6. Retain mean-pooled ChemBERTa and no-attention MLP as ablations.
7. Remove claims about atom-ring attribution unless a separate, validated attribution method maps tokens to molecular substructures.

**Acceptance criteria:**

- Attention sequence length is greater than one for ordinary multi-token SMILES.
- Padding tokens receive zero attention contribution.
- Pair-order invariance has an automated test.
- Token cross-attention must outperform the matched mean-pooled ablation on validation macro AP before it is eligible for the champion model.

### Task 9 — Add leakage-safe PrimeKG relation-aware embeddings

**Files:** new `src/features/primekg_embeddings.py`; replace or retire heuristic logic in `graph_message_passing.py`; add tests.

1. Build a drug-centered subgraph containing permitted PrimeKG biological relations such as drug-target, disease-gene, and protein-protein edges.
2. Explicitly exclude BioSNAP/Decagon DDI outcome edges and any edge derived from the target labels.
3. Map drugs by stable identifiers where available; use normalized names only as a measured fallback and report mapping coverage.
4. Train a relation-aware inductive encoder using biological attributes and edge types. It must be able to embed a held-out drug from non-DDI attributes without accessing its held-out DDI labels.
5. Fit graph normalization/statistics on training data and freeze them for validation/test.
6. Concatenate or gate these embeddings into the pair model; compare against no-PrimeKG and degree-only ablations.

**Acceptance criteria:**

- A provenance test demonstrates that no target DDI edge enters PrimeKG features.
- Coverage and fallback mapping rates appear in the manifest.
- Cold-1 and Cold-2 results are reported with and without PrimeKG embeddings.
- Documentation says “uses PrimeKG features” only if this path is enabled in the winning run.

### Task 10 — Add hierarchy consistency and validate the hierarchy

**Files:** `src/features/meddra_hierarchy.py`, `src/training/losses.py`, relevant model configs and tests.

1. Replace keyword-first SOC mapping with an authoritative CUI-to-MedDRA-SOC mapping artifact where licensing permits.
2. Keep keyword mapping only as an explicitly flagged fallback; report fallback and unmatched percentages.
3. Add hierarchy consistency loss:

   `mean(relu(p_specific - p_parent_organ))`

   and tune its weight on validation macro AP plus hierarchy-violation rate.
4. Report organ metrics both directly from the organ head and as aggregates of specific side-effect predictions.
5. Add hierarchy violation rate to `metrics.json`.

**Acceptance criteria:**

- Mapping provenance exists for every modeled CUI.
- Fallback mappings are separately auditable.
- The full model reduces validation hierarchy violations without degrading the primary metric outside its confidence interval.

### Task 11 — Multi-seed execution and final reporting

**Files:** new `src/evaluation/reporting.py`; update README/PDF generator scripts to consume result JSON.

1. Run the five fixed seeds for every required benchmark and shortlisted model.
2. Aggregate point estimates, standard deviations, and bootstrap confidence intervals.
3. Generate tables for scenario, model, label-frequency stratum, calibration, and ablation effects.
4. Make `README.md`, `MODEL_2_SEVIYE_REHBERI.md`, and both PDF generators read values from `results/benchmark_summary.json`.
5. Include benchmark ID, data hashes, sample sizes, seed count, confidence intervals, and the definition of sampled-unlabeled controls beside headline metrics.
6. Remove “cold split,” “trained on ChEMBL/PrimeKG,” “clinical reliability,” and probability claims unless the corresponding artifact proves them.

**Acceptance criteria:**

- No metric literal is duplicated manually across README, Markdown guide, and PDF generators.
- Regenerating reports from the same summary is deterministic.
- Every headline score links to a run ID and benchmark manifest.
- Reporting follows the applicable items from TRIPOD+AI and clearly states that evaluation is retrospective and non-clinical.

## 6. Dependency order

```text
Task 1
  -> Task 2
      -> Task 3
          -> Task 4
              -> Task 5
                  -> Task 6
                      -> Task 7
                          -> Task 8
                          -> Task 9
                          -> Task 10
                              -> Task 11
```

Tasks 8, 9, and 10 may proceed in parallel only after Tasks 1-7 freeze the benchmark, shared trainer, and baseline results.

## 7. Verification gates

### Gate A — Benchmark validity

Do not begin champion-model optimization until:

- split and leakage tests pass;
- train-only vocabulary selection is proven;
- negative-sampling diagnostics pass;
- the benchmark manifest is immutable and identified by hash.

### Gate B — Evaluation validity

Do not compare architectures until:

- metric fixtures pass;
- validation-only selection and calibration tests pass;
- baseline results exist for all three required scenarios.

### Gate C — Architecture promotion

Promote a component only if:

- it improves validation macro AP under matched splits and seeds;
- the improvement is not isolated to warm-start data;
- calibration and hierarchy violations do not materially worsen;
- its ablation and provenance artifacts are present.

### Gate D — Final claim

Call a model “best” only after the frozen test evaluation has completed for all five seeds. Report uncertainty and scenario-specific results; never collapse warm and cold results into one headline number.

## 8. Suggested teammate work packages

- **Package A — Benchmark and data integrity:** Tasks 1-4.
- **Package B — Metrics and trainer:** Tasks 5-7.
- **Package C — Molecular architecture:** Task 8.
- **Package D — Knowledge graph and hierarchy:** Tasks 9-10.
- **Package E — Repeated runs and documentation:** Task 11.

Packages C and D depend on A and B. Package E depends on all promoted components.

## 9. Definition of done

The roadmap is complete when:

- warm, Cold-1, and Cold-2 benchmarks exist for five seeds and pass leakage checks;
- all labels and transformations are fitted using training data only;
- sampled controls are degree matched and modeled/reported as unlabeled;
- validation selects checkpoints, thresholds, calibration, and hyperparameters;
- metrics include AP, Precision/Recall/NDCG@5, lift, calibration, operating-point statistics, uncertainty, and rare-label strata;
- token-level attention and PrimeKG/hierarchy additions have controlled ablations;
- test results are traceable to immutable configurations, data hashes, predictions, and checkpoints;
- documentation is generated from machine-readable results and makes no unsupported clinical or data-integration claims.


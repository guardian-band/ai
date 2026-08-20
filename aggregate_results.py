"""Verify run provenance and aggregate seed-level test metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from src.evaluation.metrics import compute_all_metrics
from src.models.factory import derive_run_model_id
from src.training.engine import REQUIRED_RUN_ARTIFACTS


SCALAR_METRICS = (
    "micro_ap", "micro_auroc", "brier_score", "ece_15", "precision_at_k",
    "recall_at_k", "ndcg_at_k", "macro_ap", "macro_auroc",
)
TEACHER_ABLATION_RUN_MODEL_IDS = (
    "multimodal_teacher_morgan_molformer",
    "multimodal_teacher_morgan_mpnn",
    "multimodal_teacher_morgan_hgt",
    "multimodal_teacher_full",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _artifact_digest(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def _verify_completion(run_dir: Path) -> dict[str, Any]:
    marker_path = run_dir / "completion.json"
    if not marker_path.exists():
        raise ValueError(f"run {run_dir.name} is missing completion.json")
    marker = json.loads(marker_path.read_text())
    if marker.get("status") != "complete":
        raise ValueError(f"run {run_dir.name} has non-complete completion status")
    declared = marker.get("required_artifacts")
    if not isinstance(declared, dict):
        raise ValueError(f"run {run_dir.name} has no required_artifacts provenance map")
    for artifact_name in REQUIRED_RUN_ARTIFACTS:
        path = run_dir / artifact_name
        if not path.exists():
            raise ValueError(f"run {run_dir.name} is missing required artifact {artifact_name}")
        expected = declared.get(artifact_name)
        if not isinstance(expected, dict):
            raise ValueError(f"run {run_dir.name} has no hash for {artifact_name}")
        actual = _artifact_digest(path)
        if expected.get("sha256") != actual["sha256"] or int(expected.get("bytes", -1)) != actual["bytes"]:
            raise ValueError(f"run {run_dir.name} artifact hash/size mismatch for {artifact_name}")
    # A model-specific runner may add provenance artifacts (for example the
    # Teacher validation-selection record). Validate every declared artifact,
    # not only the legacy required set, so tampering cannot be hidden behind a
    # valid core completion marker.
    for artifact_name, expected in declared.items():
        path = run_dir / artifact_name
        if not path.exists():
            raise ValueError(f"run {run_dir.name} is missing declared artifact {artifact_name}")
        if not isinstance(expected, dict):
            raise ValueError(f"run {run_dir.name} has invalid hash for {artifact_name}")
        actual = _artifact_digest(path)
        if expected.get("sha256") != actual["sha256"] or int(expected.get("bytes", -1)) != actual["bytes"]:
            raise ValueError(f"run {run_dir.name} artifact hash/size mismatch for {artifact_name}")
    return marker


def _prediction_tables(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_parquet(run_dir / "test_predictions.parquet")
    logits = pd.read_parquet(run_dir / "test_logits.parquet")
    if "pair_id" not in predictions.columns or "pair_id" not in logits.columns:
        raise ValueError(f"run {run_dir.name} prediction artifacts must contain pair_id")
    if predictions["pair_id"].isna().any() or logits["pair_id"].isna().any():
        raise ValueError(f"run {run_dir.name} prediction pair_id values must be non-null")
    if predictions["pair_id"].duplicated().any() or logits["pair_id"].duplicated().any():
        raise ValueError(f"run {run_dir.name} prediction pair_id values must be unique")
    if predictions["pair_id"].tolist() != logits["pair_id"].tolist():
        raise ValueError(f"run {run_dir.name} prediction pair order mismatch")
    truth_cols = [column for column in predictions.columns if column.startswith("truth__")]
    prob_cols = [column for column in predictions.columns if column.startswith("prob__")]
    logit_cols = [column for column in logits.columns if column.startswith("logit__")]
    labels = [column.removeprefix("truth__") for column in truth_cols]
    if not labels or len(labels) != len(set(labels)):
        raise ValueError(f"run {run_dir.name} truth labels must be unique")
    if len(prob_cols) != len(set(prob_cols)) or {column.removeprefix("prob__") for column in prob_cols} != set(labels):
        raise ValueError(f"run {run_dir.name} predictions need aligned truth__/prob__ labels")
    if {column.removeprefix("logit__") for column in logit_cols} != set(labels):
        raise ValueError(f"run {run_dir.name} logits need aligned logit__ labels")
    if predictions[truth_cols + prob_cols].isna().any().any() or logits[logit_cols].isna().any().any():
        raise ValueError(f"run {run_dir.name} prediction values must be finite")
    truth = predictions[[f"truth__{label}" for label in labels]].copy()
    truth.columns = labels
    probabilities = predictions[[f"prob__{label}" for label in labels]].copy()
    probabilities.columns = labels
    truth_values = truth.to_numpy(dtype=float)
    probability_values = probabilities.to_numpy(dtype=float)
    logit_values = logits[logit_cols].to_numpy(dtype=float)
    if not np.isfinite(truth_values).all() or not np.isfinite(probability_values).all() or not np.isfinite(logit_values).all():
        raise ValueError(f"run {run_dir.name} prediction values must be finite")
    if not np.isin(truth_values, [0.0, 1.0]).all():
        raise ValueError(f"run {run_dir.name} truth values must be binary")
    if ((probability_values < 0.0) | (probability_values > 1.0)).any():
        raise ValueError(f"run {run_dir.name} probabilities must lie in [0, 1]")
    return truth, probabilities


def _verify_run(run_dir: Path) -> dict[str, Any]:
    _verify_completion(run_dir)
    config = json.loads((run_dir / "config.resolved.json").read_text())
    if config.get("run_id") != run_dir.name:
        raise ValueError(f"run {run_dir.name} config run_id mismatch")
    for key in ("model_type", "benchmark_id", "scenario", "seed", "manifest_hash"):
        if key not in config:
            raise ValueError(f"run {run_dir.name} config missing {key}")
    model_type = str(config["model_type"])
    derived_run_model_id = derive_run_model_id(config)
    declared_run_model_id = config.get("run_model_id")
    if declared_run_model_id is not None:
        if declared_run_model_id != derived_run_model_id:
            raise ValueError(
                f"run {run_dir.name} run_model_id is not derived from model configuration"
            )
        run_model_id = str(declared_run_model_id)
    else:
        # Legacy artifacts predate run_model_id and intentionally retain their
        # model_type identity rather than being silently relabeled.
        run_model_id = model_type
    if run_dir.name.split("__", 1)[0] != run_model_id:
        raise ValueError(
            f"run {run_dir.name} model_type prefix/run_model_id prefix does not match config"
        )
    expected_prefix = (
        f"{run_model_id}__{config['scenario']}__seed_{int(config['seed'])}__{config['manifest_hash']}"
    )
    if not run_dir.name.startswith(expected_prefix):
        raise ValueError(f"run {run_dir.name} config/run identity mismatch")
    truth, probabilities = _prediction_tables(run_dir)
    thresholds = json.loads((run_dir / "thresholds.json").read_text())
    train_prevalences = config.get("train_prevalences", {})
    recomputed = compute_all_metrics(truth, probabilities, thresholds, train_prevalences)
    stored = json.loads((run_dir / "metrics.json").read_text())
    for key, value in recomputed.items():
        if not isinstance(value, (int, float, np.number)):
            continue
        stored_value = stored.get(key)
        recomputed_finite = np.isfinite(float(value))
        if not recomputed_finite:
            # JSON null is the canonical representation for a non-finite
            # evaluator result; accepting an explicit non-finite numeric value
            # keeps compatibility with legacy JSON while still rejecting a
            # finite value that disagrees with recomputation.
            if stored_value is None:
                continue
            if isinstance(stored_value, (int, float)) and not np.isfinite(float(stored_value)):
                continue
            raise ValueError(f"run {run_dir.name} metrics mismatch for {key}")
        if not isinstance(stored_value, (int, float)) or not np.isfinite(float(stored_value)):
            raise ValueError(f"run {run_dir.name} metrics mismatch for {key}")
        if abs(float(value) - float(stored_value)) > 1e-10:
            raise ValueError(f"run {run_dir.name} metrics mismatch for {key}")
    return {
        "run_id": run_dir.name,
        "run_model_id": run_model_id,
        "config": config,
        "metrics": recomputed,
    }


def _bootstrap_interval(values: list[float], *, resamples: int, seed: int, confidence: float) -> tuple[float, float]:
    numeric = np.asarray(values, dtype=float)
    if numeric.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(numeric, size=(resamples, numeric.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def _cohort_summary(records: list[dict[str, Any]], bootstrap: dict[str, Any]) -> dict[str, Any]:
    records = sorted(records, key=lambda record: int(record["config"]["seed"]))
    config = records[0]["config"]
    metric_summary: dict[str, Any] = {}
    for metric in SCALAR_METRICS:
        values = [float(record["metrics"][metric]) for record in records if np.isfinite(record["metrics"].get(metric, np.nan))]
        if not values:
            metric_summary[metric] = {"mean": None, "std": None, "lower_ci": None, "upper_ci": None}
            continue
        lower, upper = _bootstrap_interval(
            values,
            resamples=int(bootstrap.get("resamples", 2000)),
            seed=int(bootstrap.get("seed", 8675309)),
            confidence=float(bootstrap.get("confidence_level", 0.95)),
        )
        metric_summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "lower_ci": lower,
            "upper_ci": upper,
        }
    return {
        "model_type": config["model_type"],
        "run_model_id": records[0]["run_model_id"],
        "benchmark_id": config["benchmark_id"],
        "scenario": config["scenario"],
        "seeds": [int(record["config"]["seed"]) for record in records],
        "run_ids": [record["run_id"] for record in records],
        "metrics": metric_summary,
    }


def _ordered_label_hash(config: Mapping[str, Any]) -> str:
    """Read the ordered-label provenance hash from modern or transitional configs."""

    source_hashes = config.get("source_hashes")
    candidates = (
        config.get("labels_order_sha256"),
        config.get("label_order_sha256"),
        config.get("label_order_hash"),
        source_hashes.get("labels_order") if isinstance(source_hashes, dict) else None,
    )
    for value in candidates:
        if isinstance(value, str) and value:
            return value
    raise ValueError("paired comparison requires ordered label hash provenance")


def _delta_summary(
    values: list[float], *, bootstrap: Mapping[str, Any], seed_offset: int
) -> dict[str, Any]:
    numeric = np.asarray(values, dtype=float)
    if numeric.size == 0 or not np.isfinite(numeric).all():
        raise ValueError("paired comparison deltas must be finite")
    lower, upper = _bootstrap_interval(
        values,
        resamples=int(bootstrap.get("resamples", 2000)),
        seed=int(bootstrap.get("seed", 8675309)) + seed_offset,
        confidence=float(bootstrap.get("confidence_level", 0.95)),
    )
    return {
        "per_seed": [float(value) for value in values],
        "mean": float(np.mean(numeric)),
        "std": float(np.std(numeric, ddof=1)) if numeric.size > 1 else None,
        "lower_ci": lower,
        "upper_ci": upper,
    }


def _paired_ablation_comparison(
    records: list[dict[str, Any]],
    bootstrap: Mapping[str, Any],
    *,
    advanced_model_id: str = "multimodal_teacher_full",
) -> dict[str, Any]:
    """Compare one auxiliary Teacher ablation against Morgan-only."""

    baseline_id = "multimodal_teacher_morgan_only"
    advanced_id = advanced_model_id
    by_model: dict[str, dict[tuple[str, int], dict[str, Any]]] = {
        baseline_id: {},
        advanced_id: {},
    }
    for record in records:
        run_model_id = record.get("run_model_id")
        if run_model_id not in by_model:
            continue
        config = record["config"]
        key = (str(config["scenario"]), int(config["seed"]))
        if key in by_model[run_model_id]:
            raise ValueError(f"duplicate paired comparison run for {run_model_id} {key}")
        by_model[run_model_id][key] = record
    if not by_model[baseline_id] or not by_model[advanced_id]:
        raise ValueError(
            f"paired comparison requires Morgan-only and {advanced_id} Teacher runs"
        )
    if set(by_model[baseline_id]) != set(by_model[advanced_id]):
        raise ValueError("paired comparison scenario/seed sets differ")
    paired_bootstrap = dict(bootstrap)
    paired_bootstrap["confidence_level"] = 0.95

    pairs: list[dict[str, Any]] = []
    for key in sorted(by_model[baseline_id]):
        baseline = by_model[baseline_id][key]
        advanced = by_model[advanced_id][key]
        baseline_config = baseline["config"]
        advanced_config = advanced["config"]
        if baseline_config.get("manifest_hash") != advanced_config.get("manifest_hash"):
            raise ValueError("paired comparison manifest hash mismatch")
        if _ordered_label_hash(baseline_config) != _ordered_label_hash(advanced_config):
            raise ValueError("paired comparison ordered label hash mismatch")
        baseline_macro = float(baseline["metrics"]["macro_ap"])
        advanced_macro = float(advanced["metrics"]["macro_ap"])
        baseline_micro = float(baseline["metrics"]["micro_ap"])
        advanced_micro = float(advanced["metrics"]["micro_ap"])
        pairs.append(
            {
                "scenario": key[0],
                "seed": key[1],
                "morgan_macro_ap": baseline_macro,
                "advanced_macro_ap": advanced_macro,
                "delta_macro_ap": advanced_macro - baseline_macro,
                "morgan_micro_ap": baseline_micro,
                "advanced_micro_ap": advanced_micro,
                "delta_micro_ap": advanced_micro - baseline_micro,
            }
        )
    macro = _delta_summary(
        [pair["delta_macro_ap"] for pair in pairs], bootstrap=paired_bootstrap, seed_offset=0
    )
    micro = _delta_summary(
        [pair["delta_micro_ap"] for pair in pairs], bootstrap=paired_bootstrap, seed_offset=1
    )
    return {
        "baseline_run_model_id": baseline_id,
        "advanced_run_model_id": advanced_id,
        "pairs": pairs,
        "metrics": {"macro_ap": macro, "micro_ap": micro},
        "improved": bool(macro["lower_ci"] is not None and macro["lower_ci"] > 0.0),
    }


def aggregate_runs(runs_dir: str, output_file: str, benchmark_config: str | None = None) -> dict[str, Any]:
    """Verify, recompute, and aggregate complete model/scenario/seed cohorts."""

    runs_path = Path(runs_dir)
    if not runs_path.exists():
        raise ValueError(f"runs directory does not exist: {runs_dir}")
    config_path = Path(benchmark_config or "configs/benchmark.yaml")
    benchmark = yaml.safe_load(config_path.read_text())
    if not isinstance(benchmark, dict):
        raise ValueError("benchmark config must be a mapping")
    expected_seeds = [int(seed) for seed in benchmark.get("seeds", [])]
    partitions = benchmark.get("partitions")
    if not isinstance(partitions, dict):
        raise ValueError("benchmark config partitions must be a mapping")
    expected_scenarios = list(partitions.keys())
    if not expected_seeds or not expected_scenarios:
        raise ValueError("benchmark config must define seeds and partitions")
    records = [_verify_run(run_dir) for run_dir in sorted(runs_path.iterdir()) if run_dir.is_dir()]
    if not records:
        raise ValueError("no run directories found")

    benchmark_name = benchmark.get("benchmark_name")
    cohorts: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        config = record["config"]
        if config["benchmark_id"] != benchmark_name:
            raise ValueError(f"unexpected benchmark_id {config['benchmark_id']}")
        if config["scenario"] not in expected_scenarios or int(config["seed"]) not in expected_seeds:
            raise ValueError(f"unexpected scenario or seed in run {record['run_id']}")
        key = (
            str(record["run_model_id"]),
            str(config["benchmark_id"]),
            str(config["scenario"]),
        )
        cohort = cohorts.setdefault(key, [])
        if any(int(item["config"]["seed"]) == int(config["seed"]) for item in cohort):
            raise ValueError(f"duplicate run for {key} seed {config['seed']}")
        cohort.append(record)

    for key, cohort in cohorts.items():
        actual_seeds = {int(record["config"]["seed"]) for record in cohort}
        missing = set(expected_seeds) - actual_seeds
        if missing:
            raise ValueError(f"missing seeds {sorted(missing)} for cohort {key}")
        if actual_seeds != set(expected_seeds):
            raise ValueError(f"unexpected seeds for cohort {key}")

    # A model/benchmark is only comparable when it has every required
    # scenario.  Do not silently emit a partial warm-only summary.
    model_benchmarks: dict[tuple[str, str], set[str]] = {}
    for run_model_id, benchmark_id, scenario in cohorts:
        model_benchmarks.setdefault((run_model_id, benchmark_id), set()).add(scenario)
    for model_benchmark, scenarios in model_benchmarks.items():
        missing_scenarios = set(expected_scenarios) - scenarios
        if missing_scenarios:
            raise ValueError(
                f"missing scenarios {sorted(missing_scenarios)} for cohort {model_benchmark}"
            )

    bootstrap = benchmark.get("bootstrap") or {}
    summaries = [_cohort_summary(cohort, bootstrap) for _, cohort in sorted(cohorts.items())]
    if any(summary["metrics"]["macro_ap"]["mean"] is None for summary in summaries):
        raise ValueError("every required cohort must have a finite macro_ap for champion selection")
    by_model: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries:
        if summary["metrics"]["macro_ap"]["mean"] is not None:
            by_model.setdefault(summary["run_model_id"], []).append(summary)
    if not by_model:
        raise ValueError("no finite macro_ap values available for champion selection")

    model_scores: dict[str, tuple[float, float]] = {}
    for model_type, model_summaries in by_model.items():
        macro_values = [s["metrics"]["macro_ap"]["mean"] for s in model_summaries]
        micro_values = [s["metrics"]["micro_ap"]["mean"] for s in model_summaries]
        macro_mean = float(np.mean(macro_values))
        finite_micro = [float(value) for value in micro_values if value is not None and np.isfinite(value)]
        micro_mean = float(np.mean(finite_micro)) if finite_micro else float("-inf")
        model_scores[model_type] = (macro_mean, micro_mean)

    # Required deterministic order: macro mean across scenarios, micro mean,
    # then lexical model name.
    champion_model = min(
        model_scores,
        key=lambda model: (-model_scores[model][0], -model_scores[model][1], model),
    )
    champion_macro, champion_micro = model_scores[champion_model]
    champion = {
        "model_type": champion_model,
        "run_model_id": champion_model,
        "mean_macro_ap": champion_macro,
        "mean_micro_ap": None if not np.isfinite(champion_micro) else champion_micro,
    }
    paired_comparisons = []
    available_run_model_ids = {record["run_model_id"] for record in records}
    for advanced_model_id in TEACHER_ABLATION_RUN_MODEL_IDS:
        if {
            "multimodal_teacher_morgan_only",
            advanced_model_id,
        }.issubset(available_run_model_ids):
            paired_comparisons.append(
                _paired_ablation_comparison(
                    records, bootstrap, advanced_model_id=advanced_model_id
                )
            )
    output = {
        "schema_version": 1,
        "benchmark_config": str(config_path),
        "expected_seeds": expected_seeds,
        "expected_scenarios": expected_scenarios,
        "cohorts": summaries,
        "paired_ablation_comparisons": paired_comparisons,
        "champion_rule": "highest mean macro_ap averaged across required scenarios, then higher mean micro_ap, then lexicographically smaller model_type",
        "champion": champion,
    }
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False))
    rows = []
    for summary in summaries:
        row = {
            key: summary[key]
            for key in ("model_type", "run_model_id", "benchmark_id", "scenario")
        }
        row["seeds"] = ",".join(str(seed) for seed in summary["seeds"])
        row["run_ids"] = ",".join(summary["run_ids"])
        for metric, values in summary["metrics"].items():
            for stat, value in values.items():
                row[f"{metric}_{stat}"] = value
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_path.with_suffix(".csv"), index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, help="Directory containing run directories")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--benchmark-config", default="configs/benchmark.yaml")
    args = parser.parse_args()
    aggregate_runs(args.runs, args.output, args.benchmark_config)


if __name__ == "__main__":
    main()

"""Validation and tabular projection for aggregate benchmark summaries.

This module deliberately does not calculate metrics or create a second source
of truth.  It accepts only the JSON emitted by ``aggregate_results.py`` and
normalizes that document for report consumers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


REQUIRED_SCENARIOS = ("warm_pair", "cold_1", "cold_2")
DISPLAY_METRICS = (
    "macro_ap",
    "micro_ap",
    "macro_auroc",
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
)
SUMMARY_STAT_FIELDS = ("mean", "std", "lower_ci", "upper_ci")


class ReportingValidationError(ValueError):
    """Raised when an aggregation summary cannot support an evidence report."""


def _finite_or_none(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportingValidationError(f"{field} must be a numeric value or null")
    number = float(value)
    if not np.isfinite(number):
        raise ReportingValidationError(f"{field} must be finite or null")
    return number


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReportingValidationError(f"{field} must be a non-empty string")
    return value.strip()


def canonicalize_aggregation_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and deterministically normalize an aggregate-results document."""

    if not isinstance(payload, Mapping):
        raise ReportingValidationError("aggregation summary must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ReportingValidationError("schema_version must be 1")
    benchmark_config = _required_string(payload.get("benchmark_config"), "benchmark_config")

    raw_seeds = payload.get("expected_seeds")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ReportingValidationError("expected seeds must be a non-empty list")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in raw_seeds):
        raise ReportingValidationError("expected seeds must contain integers")
    expected_seeds = sorted(raw_seeds)
    if len(set(expected_seeds)) != len(expected_seeds):
        raise ReportingValidationError("expected seeds must be unique")

    raw_scenarios = payload.get("expected_scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise ReportingValidationError("expected scenarios must be a non-empty list")
    expected_scenarios = [str(item) for item in raw_scenarios]
    if len(set(expected_scenarios)) != len(expected_scenarios):
        raise ReportingValidationError("expected scenarios must be unique")
    missing_required = sorted(set(REQUIRED_SCENARIOS) - set(expected_scenarios))
    if missing_required:
        raise ReportingValidationError(f"expected scenarios missing required values: {missing_required}")
    expected_scenarios = sorted(expected_scenarios, key=lambda value: REQUIRED_SCENARIOS.index(value) if value in REQUIRED_SCENARIOS else len(REQUIRED_SCENARIOS))

    raw_cohorts = payload.get("cohorts")
    if not isinstance(raw_cohorts, list) or not raw_cohorts:
        raise ReportingValidationError("cohorts must be a non-empty list")
    cohorts: list[dict[str, Any]] = []
    identities: set[tuple[str, str, str]] = set()
    seen_run_ids: set[str] = set()
    for index, raw_cohort in enumerate(raw_cohorts):
        if not isinstance(raw_cohort, Mapping):
            raise ReportingValidationError(f"cohort {index} must be an object")
        declared_model_type = _required_string(
            raw_cohort.get("model_type"), f"cohort {index} model_type"
        )
        model_type = _required_string(
            raw_cohort.get("run_model_id", declared_model_type),
            f"cohort {index} run_model_id",
        )
        benchmark_id = _required_string(raw_cohort.get("benchmark_id"), f"cohort {index} benchmark_id")
        scenario = _required_string(raw_cohort.get("scenario"), f"cohort {index} scenario")
        if scenario not in expected_scenarios:
            raise ReportingValidationError(f"cohort {index} has unexpected scenario {scenario}")
        identity = (model_type, benchmark_id, scenario)
        if identity in identities:
            raise ReportingValidationError(f"duplicate cohort {identity}")
        identities.add(identity)

        seeds = raw_cohort.get("seeds")
        if not isinstance(seeds, list) or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
            raise ReportingValidationError(f"cohort {index} seeds must be integer list")
        if sorted(seeds) != expected_seeds:
            raise ReportingValidationError(f"cohort {index} seeds do not exactly match expected seeds")
        run_ids = raw_cohort.get("run_ids")
        if not isinstance(run_ids, list) or len(run_ids) != len(seeds) or any(not isinstance(run_id, str) or not run_id for run_id in run_ids):
            raise ReportingValidationError(f"cohort {index} run_ids must align with seeds")
        if len(set(run_ids)) != len(run_ids) or seen_run_ids.intersection(run_ids):
            raise ReportingValidationError(f"cohort {index} run_ids must be globally unique")
        for seed, run_id in zip(sorted(seeds), run_ids):
            if f"seed_{seed}" not in run_id:
                raise ReportingValidationError(f"cohort {index} run_id does not identify seed {seed}")
        seen_run_ids.update(run_ids)

        raw_metrics = raw_cohort.get("metrics")
        if not isinstance(raw_metrics, Mapping):
            raise ReportingValidationError(f"cohort {index} metrics must be an object")
        metrics: dict[str, dict[str, float | None]] = {}
        for metric in DISPLAY_METRICS:
            raw_metric = raw_metrics.get(metric)
            if not isinstance(raw_metric, Mapping):
                raise ReportingValidationError(f"cohort {index} metric {metric} is missing")
            values: dict[str, float | None] = {}
            for field in SUMMARY_STAT_FIELDS:
                if field not in raw_metric:
                    raise ReportingValidationError(f"cohort {index} metric {metric} missing {field}")
                values[field] = _finite_or_none(raw_metric[field], field=f"cohort {index} {metric}.{field}")
            metrics[metric] = values
        cohorts.append({
            "model_type": model_type,
            "benchmark_id": benchmark_id,
            "scenario": scenario,
            "seeds": expected_seeds.copy(),
            "run_ids": list(run_ids),
            "metrics": metrics,
        })

    model_benchmarks: dict[tuple[str, str], set[str]] = {}
    for cohort in cohorts:
        key = (cohort["model_type"], cohort["benchmark_id"])
        model_benchmarks.setdefault(key, set()).add(cohort["scenario"])
    for key, scenarios in model_benchmarks.items():
        missing = sorted(set(expected_scenarios) - scenarios)
        if missing:
            raise ReportingValidationError(f"model cohort {key} missing scenarios {missing}")

    champion_rule = _required_string(payload.get("champion_rule"), "champion_rule")
    raw_champion = payload.get("champion")
    if not isinstance(raw_champion, Mapping):
        raise ReportingValidationError("champion must be an object")
    champion_model = _required_string(
        raw_champion.get("run_model_id", raw_champion.get("model_type")),
        "champion run_model_id",
    )
    champion_macro = _finite_or_none(raw_champion.get("mean_macro_ap"), field="champion mean_macro_ap")
    if champion_macro is None:
        raise ReportingValidationError("champion macro_ap must be finite")
    champion_cohorts = [cohort for cohort in cohorts if cohort["model_type"] == champion_model]
    if not champion_cohorts:
        raise ReportingValidationError("champion must appear in cohorts")
    if any(cohort["metrics"]["macro_ap"]["mean"] is None for cohort in champion_cohorts):
        raise ReportingValidationError("champion cohort macro_ap must be finite")

    return {
        "schema_version": 1,
        "benchmark_config": benchmark_config,
        "expected_seeds": expected_seeds,
        "expected_scenarios": expected_scenarios,
        "cohorts": sorted(cohorts, key=lambda cohort: (cohort["model_type"], cohort["benchmark_id"], expected_scenarios.index(cohort["scenario"]))),
        "champion_rule": champion_rule,
        "champion": {
            "model_type": champion_model,
            "mean_macro_ap": champion_macro,
            "mean_micro_ap": _finite_or_none(raw_champion.get("mean_micro_ap"), field="champion mean_micro_ap"),
        },
    }


def load_aggregation_summary(path: str | Path) -> dict[str, Any]:
    """Read, validate, and canonicalize one aggregation JSON artifact."""

    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ReportingValidationError(f"aggregation summary not found: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportingValidationError(f"unable to read aggregation summary {path}: {exc}") from exc
    return canonicalize_aggregation_summary(payload)


def emit_summary_csv(summary: Mapping[str, Any], output_path: str | Path) -> Path:
    """Emit a flattened, traceable view of an already validated summary."""

    canonical = canonicalize_aggregation_summary(summary)
    rows: list[dict[str, Any]] = []
    for cohort in canonical["cohorts"]:
        row: dict[str, Any] = {
            "model_type": cohort["model_type"],
            "benchmark_id": cohort["benchmark_id"],
            "scenario": cohort["scenario"],
            "seeds": ",".join(str(seed) for seed in cohort["seeds"]),
            "run_ids": ",".join(cohort["run_ids"]),
        }
        for metric, values in cohort["metrics"].items():
            for field, value in values.items():
                row[f"{metric}_{field}"] = value
        rows.append(row)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def generate_benchmark_summary_reports(summary_path: str | Path, output_csv: str | Path | None = None) -> Path | None:
    """Compatibility entry point that only validates/projections an existing summary."""

    summary = load_aggregation_summary(summary_path)
    if output_csv is None:
        return None
    return emit_summary_csv(summary, output_csv)

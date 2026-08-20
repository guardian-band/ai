"""Aggregate staged, matched Teacher ablation comparisons.

This command is intentionally separate from ``aggregate_results.py``.  The
main aggregator requires every configured scenario and seed; this staged path
accepts an explicit scenario and seed cohort so warm-pair checkpoints can be
compared before the cold scenarios are complete.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from aggregate_results import (
    TEACHER_ABLATION_RUN_MODEL_IDS,
    _json_safe,
    _paired_ablation_comparison,
    _verify_run,
)


BASELINE_RUN_MODEL_ID = "multimodal_teacher_morgan_only"
KNOWN_RUN_MODEL_IDS = {BASELINE_RUN_MODEL_ID, *TEACHER_ABLATION_RUN_MODEL_IDS}


def _normalize_seeds(expected_seeds: Iterable[int]) -> list[int]:
    if isinstance(expected_seeds, (str, bytes)):
        raise ValueError("expected seeds must be a non-empty integer list")
    try:
        seeds = list(expected_seeds)
    except TypeError as exc:
        raise ValueError("expected seeds must be a non-empty integer list") from exc
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("expected seeds must be a non-empty integer list")
    if len(set(seeds)) != len(seeds):
        raise ValueError("expected seeds must be unique")
    return sorted(seeds)


def _normalize_variants(
    variants: Iterable[str] | None, discovered: set[str]
) -> list[str]:
    if variants is None:
        selected = [name for name in TEACHER_ABLATION_RUN_MODEL_IDS if name in discovered]
    else:
        if isinstance(variants, str):
            raise ValueError("variants must be a non-empty list")
        selected = list(variants)
        if not selected:
            raise ValueError("variants must be a non-empty list")
        if any(not isinstance(name, str) for name in selected):
            raise ValueError("variants must contain only teacher model IDs")
        if len(set(selected)) != len(selected):
            raise ValueError("variants must be unique")
    if not selected:
        raise ValueError("no teacher ablation variants were requested or discovered")
    unknown = sorted(set(selected) - set(TEACHER_ABLATION_RUN_MODEL_IDS))
    if unknown:
        raise ValueError(f"unknown teacher ablation variants: {unknown}")
    return selected


def _collect_exact_stage_records(
    records: list[dict[str, Any]],
    *,
    scenario: str,
    expected_seeds: list[int],
    run_model_id: str,
) -> list[dict[str, Any]]:
    expected_keys = {(scenario, seed) for seed in expected_seeds}
    selected = [record for record in records if record["run_model_id"] == run_model_id]
    actual_keys = [
        (str(record["config"]["scenario"]), int(record["config"]["seed"]))
        for record in selected
    ]
    if len(actual_keys) != len(set(actual_keys)):
        raise ValueError(f"duplicate {run_model_id} scenario/seed run")
    if set(actual_keys) != expected_keys:
        missing = sorted(expected_keys - set(actual_keys))
        unexpected = sorted(set(actual_keys) - expected_keys)
        raise ValueError(
            f"{run_model_id} scenario/seed set mismatch; missing={missing}, unexpected={unexpected}"
        )
    return sorted(selected, key=lambda record: int(record["config"]["seed"]))


def _comparison_csv_rows(comparison: Mapping[str, Any], scenario: str) -> list[dict[str, Any]]:
    variant = str(comparison["advanced_run_model_id"])
    metrics = comparison["metrics"]
    rows: list[dict[str, Any]] = []
    for pair in comparison["pairs"]:
        rows.append(
            {
                "row_type": "per_seed",
                "variant": variant,
                "scenario": scenario,
                "seed": pair["seed"],
                "morgan_macro_ap": pair["morgan_macro_ap"],
                "advanced_macro_ap": pair["advanced_macro_ap"],
                "delta_macro_ap": pair["delta_macro_ap"],
                "morgan_micro_ap": pair["morgan_micro_ap"],
                "advanced_micro_ap": pair["advanced_micro_ap"],
                "delta_micro_ap": pair["delta_micro_ap"],
            }
        )
    rows.append(
        {
            "row_type": "summary",
            "variant": variant,
            "scenario": scenario,
            "seed": None,
            "morgan_macro_ap": None,
            "advanced_macro_ap": None,
            "delta_macro_ap": None,
            "morgan_micro_ap": None,
            "advanced_micro_ap": None,
            "delta_micro_ap": None,
            "macro_mean": metrics["macro_ap"]["mean"],
            "macro_std": metrics["macro_ap"]["std"],
            "macro_lower_ci": metrics["macro_ap"]["lower_ci"],
            "macro_upper_ci": metrics["macro_ap"]["upper_ci"],
            "micro_mean": metrics["micro_ap"]["mean"],
            "micro_std": metrics["micro_ap"]["std"],
            "micro_lower_ci": metrics["micro_ap"]["lower_ci"],
            "micro_upper_ci": metrics["micro_ap"]["upper_ci"],
            "improved": comparison["improved"],
        }
    )
    return rows


def aggregate_ablations(
    runs_dir: str,
    output_file: str,
    *,
    scenario: str,
    expected_seeds: Iterable[int],
    variants: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Verify and aggregate one exact staged scenario/seed ablation cohort."""

    if not isinstance(scenario, str) or not scenario.strip():
        raise ValueError("scenario must be a non-empty string")
    scenario = scenario.strip()
    seeds = _normalize_seeds(expected_seeds)
    runs_path = Path(runs_dir)
    if not runs_path.is_dir():
        raise ValueError(f"runs directory does not exist: {runs_dir}")
    run_dirs = sorted(path for path in runs_path.iterdir() if path.is_dir())
    if not run_dirs:
        raise ValueError("no run directories found")
    records = [_verify_run(run_dir) for run_dir in run_dirs]
    actual_scenarios = {str(record["config"].get("scenario")) for record in records}
    if actual_scenarios != {scenario}:
        raise ValueError(
            f"staged ablation scenario mismatch: expected {scenario}, found {sorted(actual_scenarios)}"
        )
    discovered = {str(record["run_model_id"]) for record in records}
    unknown = sorted(discovered - KNOWN_RUN_MODEL_IDS)
    if unknown:
        raise ValueError(f"unexpected run_model_id values in staged cohort: {unknown}")
    if BASELINE_RUN_MODEL_ID not in discovered:
        raise ValueError("Morgan-only baseline is required for staged ablation comparison")
    baseline_records = _collect_exact_stage_records(
        records,
        scenario=scenario,
        expected_seeds=seeds,
        run_model_id=BASELINE_RUN_MODEL_ID,
    )
    selected_variants = _normalize_variants(variants, discovered)
    comparisons: list[dict[str, Any]] = []
    for variant in selected_variants:
        variant_records = _collect_exact_stage_records(
            records,
            scenario=scenario,
            expected_seeds=seeds,
            run_model_id=variant,
        )
        comparison_records = baseline_records + variant_records
        comparisons.append(
            _paired_ablation_comparison(
                comparison_records,
                {"resamples": 2000, "seed": 8675309, "confidence_level": 0.95},
                advanced_model_id=variant,
            )
        )
    output = {
        "schema_version": 1,
        "scenario": scenario,
        "expected_seeds": seeds,
        "baseline_run_model_id": BASELINE_RUN_MODEL_ID,
        "variants": selected_variants,
        "comparisons": comparisons,
    }
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False))
    rows = [
        row
        for comparison in comparisons
        for row in _comparison_csv_rows(comparison, scenario)
    ]
    pd.DataFrame(rows).to_csv(output_path.with_suffix(".csv"), index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, help="Directory containing completed run directories")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--scenario", required=True, help="One exact benchmark scenario, e.g. warm_pair")
    parser.add_argument("--seeds", required=True, nargs="+", type=int)
    parser.add_argument(
        "--variants",
        nargs="*",
        help="Optional teacher ablation IDs; omitted means discover all present variants",
    )
    args = parser.parse_args()
    aggregate_ablations(
        args.runs,
        args.output,
        scenario=args.scenario,
        expected_seeds=args.seeds,
        variants=args.variants,
    )


if __name__ == "__main__":
    main()

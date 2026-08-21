#!/usr/bin/env python3
"""Run the no-training gate before any pair-conditioned cold-start model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.cold_start_feasibility import (  # noqa: E402
    audit_safe_paths,
    evaluate_similarity_transfer,
    load_manifest_tables,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--primekg", type=Path, required=True)
    parser.add_argument("--morgan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-morgan-macro-auprc", type=float)
    args = parser.parse_args()

    manifest, pairs, triples, labels = load_manifest_tables(args.manifest)
    primekg = pd.read_parquet(args.primekg) if args.primekg.suffix == ".parquet" else pd.read_csv(args.primekg)
    morgan = pd.read_parquet(args.morgan)
    path_report = audit_safe_paths(primekg, pairs, labels)
    similarity = evaluate_similarity_transfer(pairs, triples, labels, morgan)
    reference = args.reference_morgan_macro_auprc
    similarity_delta = None if reference is None else similarity.report["macro_auprc"] - reference
    val_coverage = path_report["splits"]["validation"]["coverage"]
    no_signal = val_coverage == 0.0 and similarity_delta is not None and similarity_delta <= 0.0
    report = {
        "scenario": manifest.get("scenario"),
        "seed": manifest.get("seed"),
        "selection_policy": {
            "uses_test_for_model_selection": False,
            "promotion_delta_macro_auprc": 0.002,
            "no_go_only_if_no_safe_paths_and_similarity_does_not_beat_morgan": True,
        },
        "safe_path_audit": path_report,
        "similarity_transfer": similarity.report,
        "reference_morgan_validation_macro_auprc": reference,
        "similarity_delta_vs_reference": similarity_delta,
        "recommendation": "stop_expensive_path_model" if no_signal else "pair_conditioned_teacher_is_feasible",
        "limitations": [
            "Similarity k is selected on validation and is exploratory, not a final test estimate.",
            "Temporal evaluation requires a separately sourced approval-date artifact.",
            "Text features remain disabled until a leakage-sanitized source is available.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"FEASIBILITY_COMPLETE: {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Restore Cold-1 artifacts and run only the cheap directed path probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_advanced_colab_sweep import _restore, ensure_manifest_hierarchy  # noqa: E402
from scripts.run_colab_shared_baseline_race import _python, _required, _run_stage  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-backup", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    scenario, seed = "cold_1", 42
    artifacts = ROOT / "artifacts"
    _restore(args.source_backup)
    _restore(args.drive_root / f"{scenario}_seed_{seed}")
    manifest = _required(
        artifacts / "benchmarks/polypharmacy_v2/cold_1/seed_42/manifest.json",
        "Cold-1 manifest",
    )
    hierarchy = _required(artifacts / "meddra_hierarchy.json", "MedDRA hierarchy")
    ensure_manifest_hierarchy(hierarchy, manifest)
    manifest_hash = str(json.loads(manifest.read_text())["manifest_hash"])
    config = _required(
        artifacts / "configs/multimodal_teacher_morgan_only_cold_1_seed_42.yaml",
        "Morgan-only experiment config",
    )
    run_dir = artifacts / "runs" / (
        f"multimodal_teacher_morgan_only__cold_1__seed_42__{manifest_hash}"
    )
    baseline = _required(run_dir / "checkpoint_baseline.pt", "frozen Morgan checkpoint")
    primekg = _required(
        ROOT / "data/processed/colab_primekg_safe.parquet", "safe PrimeKG parquet"
    )
    report = artifacts / "reports/cold_1_seed_42_directed_path_probe.json"
    backup = args.drive_root / f"{scenario}_seed_{seed}"
    _run_stage(
        label="directed_path_feature_probe_morgan_residual_v3",
        command=_python(
            "scripts/probe_cold_path_features.py",
            "--experiment", str(config),
            "--manifest", str(manifest),
            "--primekg", str(primekg),
            "--baseline-checkpoint", str(baseline),
            "--output", str(report),
            "--permutations", "500",
        ),
        outputs=[report],
        backup=backup,
        scenario=scenario,
        seed=seed,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        result = json.loads(report.read_text())
        print("COLD_PATH_PROBE_COMPLETE", flush=True)
        print(f"baseline={result['baseline_validation_macro_auprc']:.9f}", flush=True)
        print(f"corrected={result['best_correction_validation_macro_auprc']:.9f}", flush=True)
        print(f"delta={result['delta']:+.9f}", flush=True)
        print(f"decision={result['decision']}", flush=True)


if __name__ == "__main__":
    main()

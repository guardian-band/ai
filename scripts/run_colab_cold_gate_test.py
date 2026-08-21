"""Run only the Cold-1 seed-42 MPNN fused stage with gate logit -2."""

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
    if not args.dry_run:
        _restore(args.source_backup)
        _restore(args.drive_root / f"{scenario}_seed_{seed}")

    manifest = _required(
        artifacts / "benchmarks/polypharmacy_v2/cold_1/seed_42/manifest.json",
        "Cold-1 seed-42 manifest",
    )
    hierarchy = _required(artifacts / "meddra_hierarchy.json", "MedDRA hierarchy")
    if not args.dry_run:
        ensure_manifest_hierarchy(hierarchy, manifest)
    manifest_hash = str(json.loads(manifest.read_text())["manifest_hash"])
    features = _required(
        artifacts / "features/advanced_cold_1_seed_42.npz",
        "assembled Cold-1 features",
    )
    baseline_runs = sorted(
        (artifacts / "runs").glob(
            f"multimodal_teacher_morgan_only__cold_1__seed_42__{manifest_hash}"
        )
    )
    if not baseline_runs:
        raise FileNotFoundError("shared Cold-1 Morgan baseline run is missing")
    baseline = _required(baseline_runs[-1] / "checkpoint_baseline.pt", "shared Morgan checkpoint")

    variant = "multimodal_teacher_morgan_mpnn_gate_m2"
    config = artifacts / "configs" / f"{variant}_cold_1_seed_42.yaml"
    run_dir = artifacts / "runs" / f"{variant}__cold_1__seed_42__{manifest_hash}"
    backup = args.drive_root / "cold_1_seed_42"
    _run_stage(
        label="config_gate_m2",
        command=_python(
            "configure_advanced_experiment.py",
            "--model", "teacher",
            "--template", str(ROOT / "configs/ablations/multimodal_teacher_morgan_mpnn_gate_m2.yaml"),
            "--manifest", str(manifest),
            "--features", str(features),
            "--hierarchy", str(hierarchy),
            "--output", str(config),
            "--baseline-checkpoint", str(baseline),
        ),
        outputs=[config], backup=backup, scenario=scenario, seed=seed,
        dry_run=args.dry_run,
    )
    _run_stage(
        label="train_gate_m2",
        command=_python(
            "run_precomputed_experiment.py",
            "--experiment", str(config),
            "--benchmark", str(manifest),
        ),
        outputs=[run_dir], backup=backup, scenario=scenario, seed=seed,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        selection = json.loads((run_dir / "teacher_validation_selection.json").read_text())
        baseline_ap = float(selection["baseline"]["macro_ap"])
        fused_ap = float(selection["combined"]["macro_ap"])
        print("GATE_TEST_COMPLETE", flush=True)
        print(f"baseline_val_macro_ap={baseline_ap:.9f}", flush=True)
        print(f"fused_val_macro_ap={fused_ap:.9f}", flush=True)
        print(f"delta={fused_ap - baseline_ap:+.9f}", flush=True)
        print(f"selected={selection['selected']['mode']}", flush=True)


if __name__ == "__main__":
    main()

"""Run the economical warm-pair Teacher race with one shared Morgan baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_advanced_colab_sweep import (  # noqa: E402
    _backup_stage,
    _gpu_status,
    _marker_valid,
    _restore,
    ensure_manifest_hierarchy,
)


VARIANTS = (
    "multimodal_teacher_morgan_only",
    "multimodal_teacher_morgan_mpnn",
    "multimodal_teacher_morgan_hgt",
)
OPTIONAL_VARIANTS = VARIANTS[1:]


def _python(entrypoint: str, *args: str) -> list[str]:
    return [sys.executable, "-u", str(ROOT / entrypoint), *args]


def _run_live(command: list[str], label: str, dry_run: bool) -> None:
    print(f"RUN {label}: {' '.join(command)}", flush=True)
    if dry_run:
        return
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=ROOT)
    while True:
        try:
            code = process.wait(timeout=30)
            break
        except subprocess.TimeoutExpired:
            elapsed = (time.monotonic() - started) / 60
            print(
                f"HEARTBEAT {label}: {elapsed:.1f} min | "
                f"GPU %, used MB, total MB: {_gpu_status()}",
                flush=True,
            )
    if code:
        raise subprocess.CalledProcessError(code, command)


def _run_stage(
    *,
    label: str,
    command: list[str],
    outputs: list[Path],
    backup: Path,
    seed: int,
    dry_run: bool,
) -> None:
    marker = backup / "markers" / f"{label}.complete.json"
    if not dry_run and _marker_valid(marker):
        print(f"SKIP verified: warm_pair/seed_{seed}/{label}", flush=True)
        return
    _run_live(command, f"warm_pair/seed_{seed}/{label}", dry_run)
    if dry_run:
        return
    _backup_stage(backup, marker, outputs, "warm_pair", seed, label)
    if not _marker_valid(marker):
        raise RuntimeError(f"Drive backup verification failed: {label}")
    print(f"SAVED+VERIFIED: warm_pair/seed_{seed}/{label}", flush=True)


def _required(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def _run_seed(
    seed: int, drive_root: Path, dry_run: bool, variants: tuple[str, ...]
) -> list[Path]:
    artifacts = ROOT / "artifacts"
    backup = drive_root / f"warm_pair_seed_{seed}"
    if not dry_run:
        _restore(backup)

    manifest = artifacts / (
        f"benchmarks/polypharmacy_v2/warm_pair/seed_{seed}/manifest.json"
    )
    hierarchy = artifacts / "meddra_hierarchy.json"
    _required(manifest, "benchmark manifest")
    _required(hierarchy, "hierarchy")
    manifest_payload = json.loads(manifest.read_text())
    manifest_hash = str(manifest_payload["manifest_hash"])
    if not dry_run:
        ensure_manifest_hierarchy(hierarchy, manifest)

    molformer = _required(
        artifacts / "features/molformer_tokens.npz", "MolFormer features"
    )
    mpnn = _required(
        artifacts / "features/mpnn_warm_pair_seed_42_lossless.npz",
        "lossless MPNN features",
    )
    hgt_checkpoint = artifacts / "checkpoints" / f"hgt_warm_pair_seed_{seed}.pt"
    hgt = artifacts / "features" / f"hgt_warm_pair_seed_{seed}.npz"
    features = artifacts / "features" / f"advanced_warm_pair_seed_{seed}.npz"

    if "multimodal_teacher_morgan_hgt" in variants:
        _run_stage(
            label="hgt",
            command=_python(
                "scripts/train_primekg_hgt.py",
                "--primekg",
                "data/processed/colab_primekg_safe.parquet",
                "--morgan",
                "artifacts/morgan_fingerprints.parquet",
                "--manifest",
                str(manifest),
                "--checkpoint",
                str(hgt_checkpoint),
                "--output",
                str(hgt),
                "--hidden-dim",
                "128",
                "--batch-size",
                "1024",
                "--num-neighbors",
                "5",
                "3",
                "2",
                "--epochs",
                "5",
                "--max-train-edges-per-relation",
                "5000",
                "--max-validation-edges-per-relation",
                "1000",
                "--seed",
                str(seed),
                "--device",
                "cuda",
            ),
            outputs=[
                hgt_checkpoint,
                hgt_checkpoint.with_name(f"{hgt_checkpoint.name}.sha256"),
                hgt_checkpoint.with_name(
                    f"{hgt_checkpoint.name}.training_metrics.json"
                ),
                hgt_checkpoint.with_name(
                    f"{hgt_checkpoint.name}.training_metrics.json.sha256"
                ),
                hgt,
                hgt.with_suffix(hgt.suffix + ".sha256"),
            ],
            backup=backup,
            seed=seed,
            dry_run=dry_run,
        )
    assemble_command = _python(
        "build_advanced_features.py",
        "--manifest",
        str(manifest),
        "--morgan",
        "artifacts/morgan_fingerprints.parquet",
        "--molformer",
        str(molformer),
        "--mpnn",
        str(mpnn),
        "--output",
        str(features),
    )
    if "multimodal_teacher_morgan_hgt" in variants:
        assemble_command.extend(["--kg", str(hgt)])
    _run_stage(
        label="assemble",
        command=assemble_command,
        outputs=[features, features.with_suffix(features.suffix + ".sha256")],
        backup=backup,
        seed=seed,
        dry_run=dry_run,
    )

    completed_runs: list[Path] = []
    shared_baseline: Path | None = None
    for variant in (VARIANTS[0], *variants):
        template = ROOT / "configs/ablations" / f"{variant}.yaml"
        config = artifacts / "configs" / f"{variant}_warm_pair_seed_{seed}.yaml"
        run_dir = artifacts / "runs" / (
            f"{variant}__warm_pair__seed_{seed}__{manifest_hash}"
        )
        configure = _python(
            "configure_advanced_experiment.py",
            "--model", "teacher",
            "--template", str(template),
            "--manifest", str(manifest),
            "--features", str(features),
            "--hierarchy", str(hierarchy),
            "--output", str(config),
        )
        if shared_baseline is not None:
            configure.extend(["--baseline-checkpoint", str(shared_baseline)])
        _run_stage(
            label=f"config_{variant}",
            command=configure,
            outputs=[config],
            backup=backup,
            seed=seed,
            dry_run=dry_run,
        )
        _run_stage(
            label=f"train_{variant}",
            command=_python(
                "run_precomputed_experiment.py",
                "--experiment", str(config),
                "--benchmark", str(manifest),
            ),
            outputs=[run_dir],
            backup=backup,
            seed=seed,
            dry_run=dry_run,
        )
        completed_runs.append(run_dir)
        if variant == "multimodal_teacher_morgan_only":
            shared_baseline = run_dir / "checkpoint_baseline.pt"
            if not dry_run:
                _required(shared_baseline, "shared Morgan baseline checkpoint")
    return completed_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 2024])
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=OPTIONAL_VARIANTS,
        default=list(OPTIONAL_VARIANTS),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be a non-empty unique list")
    variants = tuple(dict.fromkeys(args.variants))

    print("WARM-PAIR SHARED-BASELINE RACE", flush=True)
    print(f"Seeds: {args.seeds}", flush=True)
    print(f"Models: Morgan-only, {', '.join(variants)}", flush=True)
    all_runs: list[Path] = []
    for seed in args.seeds:
        all_runs.extend(_run_seed(seed, args.drive_root, args.dry_run, variants))
    if args.dry_run:
        return

    cohort = ROOT / "artifacts" / "race_runs" / "warm_pair_shared_baseline"
    if cohort.exists():
        shutil.rmtree(cohort)
    cohort.mkdir(parents=True)
    for run_dir in all_runs:
        shutil.copytree(run_dir, cohort / run_dir.name)
    report = ROOT / "artifacts" / "reports" / "warm_pair_shared_baseline_race.json"
    _run_live(
        _python(
            "aggregate_ablations.py",
            "--runs", str(cohort),
            "--output", str(report),
            "--scenario", "warm_pair",
            "--seeds", *(str(seed) for seed in args.seeds),
            "--variants",
            *variants,
        ),
        "aggregate_shared_baseline_race",
        False,
    )
    report_backup = args.drive_root / "reports"
    report_backup.mkdir(parents=True, exist_ok=True)
    shutil.copy2(report, report_backup / report.name)
    shutil.copy2(report.with_suffix(".csv"), report_backup / report.with_suffix(".csv").name)
    print(f"COMPLETE: {args.drive_root}", flush=True)


if __name__ == "__main__":
    main()

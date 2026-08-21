"""Run the matched warm-pair Teacher ablations with resumable Drive backups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml


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
from src.models.factory import derive_run_model_id  # noqa: E402


VARIANTS = (
    "multimodal_teacher_morgan_only",
    "multimodal_teacher_morgan_molformer",
    "multimodal_teacher_morgan_mpnn",
    "multimodal_teacher_morgan_hgt",
    "multimodal_teacher_full",
)


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
    dry_run: bool,
) -> None:
    marker = backup / "markers" / f"{label}.complete.json"
    if not dry_run and _marker_valid(marker):
        print(f"SKIP verified: {label}", flush=True)
        return
    _run_live(command, label, dry_run)
    if dry_run:
        return
    _backup_stage(backup, marker, outputs, "warm_pair", 42, label)
    if not _marker_valid(marker):
        raise RuntimeError(f"Drive backup verification failed: {label}")
    print(f"SAVED+VERIFIED: {label}", flush=True)


def _python(entrypoint: str, *args: str) -> list[str]:
    return [sys.executable, "-u", str(ROOT / entrypoint), *args]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scenario, seed = "warm_pair", 42
    backup = args.drive_root / f"{scenario}_seed_{seed}"
    if not args.dry_run:
        _restore(backup)

    artifacts = ROOT / "artifacts"
    manifest = artifacts / "benchmarks/polypharmacy_v2/warm_pair/seed_42/manifest.json"
    hierarchy = artifacts / "meddra_hierarchy.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing benchmark manifest: {manifest}")
    if not hierarchy.is_file():
        raise FileNotFoundError(f"missing hierarchy: {hierarchy}")
    manifest_payload = json.loads(manifest.read_text())
    manifest_hash = str(manifest_payload["manifest_hash"])
    if not args.dry_run:
        ensure_manifest_hierarchy(hierarchy, manifest)

    molformer = artifacts / "features/molformer_tokens.npz"
    mpnn_checkpoint = artifacts / "checkpoints/molecular_mpnn.pt"
    mpnn = artifacts / "features/mpnn_tokens.npz"
    hgt_checkpoint = artifacts / "checkpoints/hgt_warm_pair_seed_42.pt"
    hgt = artifacts / "features/hgt_warm_pair_seed_42.npz"
    features = artifacts / "features/advanced_warm_pair_seed_42.npz"

    shared = (
        (
            "molformer",
            _python(
                "scripts/extract_molformer_tokens.py",
                "--input", "data/raw/drugs_master.csv",
                "--output", str(molformer),
                "--revision", "361063d0ad524ef77cf39b08469f6be770dc550f",
                "--allow-remote-code", "--device", "cuda",
            ),
            [molformer, molformer.with_suffix(molformer.suffix + ".sha256")],
        ),
        (
            "mpnn_train",
            _python(
                "scripts/train_molecular_mpnn.py",
                "--input", "data/raw/drugs_master.csv",
                "--checkpoint", str(mpnn_checkpoint), "--device", "cuda",
            ),
            [
                mpnn_checkpoint,
                mpnn_checkpoint.with_name(f"{mpnn_checkpoint.name}.sha256"),
            ],
        ),
        (
            "mpnn_export",
            _python(
                "scripts/export_molecular_mpnn_tokens.py",
                "--input", "data/raw/drugs_master.csv",
                "--checkpoint", str(mpnn_checkpoint),
                "--output", str(mpnn), "--device", "cuda",
            ),
            [mpnn, mpnn.with_suffix(mpnn.suffix + ".sha256")],
        ),
        (
            "hgt",
            _python(
                "scripts/train_primekg_hgt.py",
                "--primekg", "data/processed/colab_primekg_safe.parquet",
                "--morgan", "artifacts/morgan_fingerprints.parquet",
                "--manifest", str(manifest),
                "--checkpoint", str(hgt_checkpoint), "--output", str(hgt),
                "--hidden-dim", "128", "--batch-size", "1024",
                "--num-neighbors", "5", "3", "2", "--epochs", "5",
                "--max-train-edges-per-relation", "5000",
                "--max-validation-edges-per-relation", "1000",
                "--seed", "42", "--device", "cuda",
            ),
            [
                hgt_checkpoint,
                hgt_checkpoint.with_name(f"{hgt_checkpoint.name}.sha256"),
                hgt_checkpoint.with_name(f"{hgt_checkpoint.name}.training_metrics.json"),
                hgt_checkpoint.with_name(
                    f"{hgt_checkpoint.name}.training_metrics.json.sha256"
                ),
                hgt,
                hgt.with_suffix(hgt.suffix + ".sha256"),
            ],
        ),
        (
            "assemble",
            _python(
                "build_advanced_features.py", "--manifest", str(manifest),
                "--morgan", "artifacts/morgan_fingerprints.parquet",
                "--molformer", str(molformer), "--mpnn", str(mpnn),
                "--kg", str(hgt), "--output", str(features),
            ),
            [features, features.with_suffix(features.suffix + ".sha256")],
        ),
    )
    for label, command, outputs in shared:
        _run_stage(
            label=label, command=command, outputs=outputs,
            backup=backup, dry_run=args.dry_run,
        )

    for variant in VARIANTS:
        template = ROOT / "configs/ablations" / f"{variant}.yaml"
        payload = yaml.safe_load(template.read_text())
        if derive_run_model_id(payload) != variant:
            raise ValueError(f"ablation template/model ID mismatch: {variant}")
        config = artifacts / "configs" / f"{variant}_warm_pair_seed_42.yaml"
        run_dir = artifacts / "runs" / f"{variant}__warm_pair__seed_42__{manifest_hash}"
        _run_stage(
            label=f"config_{variant}",
            command=_python(
                "configure_advanced_experiment.py", "--model", "teacher",
                "--template", str(template), "--manifest", str(manifest),
                "--features", str(features), "--hierarchy", str(hierarchy),
                "--output", str(config),
            ),
            outputs=[config], backup=backup, dry_run=args.dry_run,
        )
        _run_stage(
            label=f"train_{variant}",
            command=_python(
                "run_precomputed_experiment.py", "--experiment", str(config),
                "--benchmark", str(manifest),
            ),
            outputs=[run_dir], backup=backup, dry_run=args.dry_run,
        )

    report = artifacts / "reports/warm_pair_seed_42_ablations.json"
    _run_stage(
        label="aggregate",
        command=_python(
            "aggregate_ablations.py", "--runs", str(artifacts / "runs"),
            "--output", str(report), "--scenario", "warm_pair", "--seeds", "42",
        ),
        outputs=[report, report.with_suffix(".csv")],
        backup=backup, dry_run=args.dry_run,
    )
    print(f"COMPLETE: results are stored under {backup}", flush=True)


if __name__ == "__main__":
    main()

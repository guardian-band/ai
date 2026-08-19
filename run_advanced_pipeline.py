"""Run configured advanced-pipeline stages sequentially with resumable boundaries."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parent
ENTRYPOINTS = {
    "molformer": ROOT / "scripts/extract_molformer_tokens.py",
    "mpnn_train": ROOT / "scripts/train_molecular_mpnn.py",
    "mpnn_export": ROOT / "scripts/export_molecular_mpnn_tokens.py",
    "hgt": ROOT / "scripts/train_primekg_hgt.py",
    "assemble": ROOT / "build_advanced_features.py",
    "teacher_config": ROOT / "configure_advanced_experiment.py",
    "teacher": ROOT / "run_precomputed_experiment.py",
    "cache": ROOT / "build_teacher_cache.py",
    "student_config": ROOT / "configure_advanced_experiment.py",
    "student": ROOT / "run_precomputed_experiment.py",
}


def _jobs(value, stage):
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [value]
    if (
        isinstance(value, list)
        and value
        and all(
            isinstance(job, list) and all(isinstance(item, str) for item in job)
            for job in value
        )
    ):
        return value
    raise ValueError(f"pipeline stage {stage} must be an argv list or list of argv lists")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--start-at", choices=tuple(ENTRYPOINTS))
    parser.add_argument("--stop-after", choices=tuple(ENTRYPOINTS))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = yaml.safe_load(args.config.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("stages"), dict):
        raise ValueError("pipeline config must contain a stages mapping")
    configured = payload["stages"]
    unknown = sorted(set(configured) - set(ENTRYPOINTS))
    if unknown:
        raise ValueError(f"unknown pipeline stages: {', '.join(unknown)}")
    names = list(ENTRYPOINTS)
    start = names.index(args.start_at) if args.start_at else 0
    stop = names.index(args.stop_after) if args.stop_after else len(names) - 1
    if start > stop:
        raise ValueError("start-at must not come after stop-after")
    for stage in names[start : stop + 1]:
        if stage not in configured:
            continue
        for job_number, job_args in enumerate(_jobs(configured[stage], stage), start=1):
            command = [sys.executable, str(ENTRYPOINTS[stage]), *job_args]
            print(f"[{stage} {job_number}] {' '.join(command)}", flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

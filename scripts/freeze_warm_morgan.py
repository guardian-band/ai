#!/usr/bin/env python3
"""Freeze the paired Morgan baseline from an already frozen warm Teacher run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def freeze_paired_morgan(teacher_release: Path, output: Path) -> dict[str, Any]:
    teacher_manifest_path = teacher_release / "frozen_teacher_manifest.json"
    teacher_run = teacher_release / "run"
    selection_path = teacher_run / "teacher_validation_selection.json"
    baseline_checkpoint = teacher_run / "checkpoint_baseline.pt"
    required = (teacher_manifest_path, selection_path, baseline_checkpoint)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen Teacher provenance files: {missing}")

    teacher_manifest = json.loads(teacher_manifest_path.read_text())
    selection = json.loads(selection_path.read_text())
    winner = teacher_manifest["winner"]
    baseline = selection["baseline"]

    output.mkdir(parents=True, exist_ok=True)
    destination = output / "checkpoint_best.pt"
    source_hash = _sha256(baseline_checkpoint)
    if destination.exists():
        if _sha256(destination) != source_hash:
            raise ValueError("frozen Morgan checkpoint already exists with different provenance")
    else:
        temporary = output / ".checkpoint_best.pt.tmp"
        shutil.copy2(baseline_checkpoint, temporary)
        os.replace(temporary, destination)

    manifest = {
        "schema_version": 1,
        "model_family": "Morgan-only pair predictor",
        "release_role": "CPU product candidate after Student retention rejection",
        "selection_basis": "paired baseline from the validation-selected frozen warm Morgan+MPNN Teacher",
        "test_metrics_used_for_selection": False,
        "scenario": str(winner["scenario"]),
        "seed": int(winner["seed"]),
        "validation_macro_ap": float(baseline["macro_ap"]),
        "source_teacher_validation_macro_ap": float(winner["validation_macro_ap"]),
        "source_teacher_release_manifest_sha256": _sha256(teacher_manifest_path),
        "source_teacher_selection_sha256": _sha256(selection_path),
        "source_baseline_checkpoint_sha256": source_hash,
        "frozen_checkpoint_sha256": _sha256(destination),
        "frozen_checkpoint": str(destination.resolve()),
    }
    _atomic_json(output / "frozen_morgan_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze_paired_morgan(args.teacher_release, args.output)
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"WARM_MORGAN_FROZEN: {args.output}", flush=True)


if __name__ == "__main__":
    main()

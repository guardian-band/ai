#!/usr/bin/env python3
"""Audit completed warm Morgan+MPNN runs and freeze the validation winner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aggregate_results import _verify_run


MODEL_ID = "multimodal_teacher_morgan_mpnn"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_warm_teacher(candidates: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Select highest validation AP with deterministic seed/hash tie-breaking."""

    eligible = [
        dict(item)
        for item in candidates
        if item.get("scenario") == "warm_pair"
        and item.get("run_model_id") == MODEL_ID
        and item.get("selected_mode") == "fused"
    ]
    if not eligible:
        raise ValueError("no completed warm Morgan+MPNN run selected fused on validation")
    return sorted(
        eligible,
        key=lambda item: (
            -float(item["validation_macro_ap"]),
            int(item["seed"]),
            str(item["checkpoint_sha256"]),
        ),
    )[0]


def _candidate(run_dir: Path) -> dict[str, Any]:
    verified = _verify_run(run_dir)
    config = verified["config"]
    selection_path = run_dir / "teacher_validation_selection.json"
    selection = json.loads(selection_path.read_text())
    checkpoint = run_dir / "checkpoint_best.pt"
    return {
        "run_dir": str(run_dir.resolve()),
        "run_id": str(config["run_id"]),
        "run_model_id": str(verified["run_model_id"]),
        "scenario": str(config["scenario"]),
        "seed": int(config["seed"]),
        "manifest_hash": str(config["manifest_hash"]),
        "selected_mode": str(selection["selected"]["mode"]),
        "validation_macro_ap": float(selection["selected"]["macro_ap"]),
        "baseline_validation_macro_ap": float(selection["baseline"]["macro_ap"]),
        "fused_validation_macro_ap": float(selection["combined"]["macro_ap"]),
        "checkpoint_sha256": _sha256(checkpoint),
        "selection_sha256": _sha256(selection_path),
        "completion_sha256": _sha256(run_dir / "completion.json"),
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pattern = f"{MODEL_ID}__warm_pair__seed_*__*"
    discovered = sorted(path for path in args.search_root.rglob(pattern) if path.is_dir())
    if not discovered:
        raise FileNotFoundError(f"no warm Morgan+MPNN runs under {args.search_root}")

    candidates_by_checkpoint: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, str]] = []
    for run_dir in discovered:
        try:
            candidate = _candidate(run_dir)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            rejected.append({"run_dir": str(run_dir), "reason": str(exc)})
            continue
        candidates_by_checkpoint.setdefault(candidate["checkpoint_sha256"], candidate)
    candidates = list(candidates_by_checkpoint.values())
    winner = select_warm_teacher(candidates)

    frozen_run = args.output / "run"
    source = Path(winner["run_dir"])
    if frozen_run.exists():
        existing_checkpoint = frozen_run / "checkpoint_best.pt"
        if not existing_checkpoint.is_file() or _sha256(existing_checkpoint) != winner["checkpoint_sha256"]:
            raise ValueError("frozen Teacher already exists with different provenance")
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        temporary = args.output / ".run.tmp"
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.copytree(source, temporary)
        os.replace(temporary, frozen_run)

    manifest = {
        "schema_version": 1,
        "selection_basis": "highest validation Macro AUPRC among completed warm Morgan+MPNN runs whose validation selection chose fused",
        "test_metrics_used_for_selection": False,
        "winner": {**winner, "frozen_run_dir": str(frozen_run.resolve())},
        "candidates": sorted(candidates, key=lambda item: (int(item["seed"]), item["checkpoint_sha256"])),
        "rejected": rejected,
    }
    _atomic_json(args.output / "frozen_teacher_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"WARM_TEACHER_FROZEN: {args.output}", flush=True)


if __name__ == "__main__":
    main()

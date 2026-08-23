#!/usr/bin/env python3
"""Build a self-contained, hash-verified model directory for the HTTP service."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        help="Optional verified ONNX directory containing both graphs and metadata.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    experiment = args.experiment.resolve()
    manifest = args.manifest.resolve()
    checkpoint = args.checkpoint.resolve()
    selection = args.selection.resolve()
    onnx_dir = args.onnx_dir.resolve() if args.onnx_dir else None
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(experiment.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("model_type") != "multimodal_teacher":
        raise ValueError("experiment must declare model_type=multimodal_teacher")

    feature_path = Path(str(config["feature_artifact_path"]))
    hierarchy_path = Path(str(config["hierarchy_path"]))
    if not feature_path.is_absolute():
        feature_path = (experiment.parent / feature_path).resolve()
    if not hierarchy_path.is_absolute():
        hierarchy_path = (experiment.parent / hierarchy_path).resolve()

    temporary = Path(tempfile.mkdtemp(prefix="polypharmacy-model-release-", dir=output.parent))
    try:
        _copy(feature_path, temporary / "feature_artifact.npz")
        feature_checksum = feature_path.with_name(f"{feature_path.name}.sha256")
        if feature_checksum.is_file():
            _copy(feature_checksum, temporary / "feature_artifact.npz.sha256")
        _copy(hierarchy_path, temporary / "meddra_hierarchy.json")
        _copy(checkpoint, temporary / "checkpoint_best.pt")
        _copy(selection, temporary / "teacher_validation_selection.json")
        shutil.copytree(manifest.parent, temporary / "benchmark", dirs_exist_ok=True)
        if onnx_dir is not None:
            for name in (
                "teacher_fused.onnx",
                "teacher_baseline.onnx",
                "onnx_release.json",
            ):
                _copy(onnx_dir / name, temporary / "onnx" / name)

        config["feature_artifact_path"] = "feature_artifact.npz"
        config["hierarchy_path"] = "meddra_hierarchy.json"
        (temporary / "teacher.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )

        metadata = {
            "schema_version": 1,
            "model": "warm_morgan_mpnn_teacher",
            "experiment": "teacher.yaml",
            "manifest": "benchmark/manifest.json",
            "checkpoint": "checkpoint_best.pt",
            "selection": "teacher_validation_selection.json",
            "sha256": {
                "checkpoint": _sha256(temporary / "checkpoint_best.pt"),
                "selection": _sha256(temporary / "teacher_validation_selection.json"),
                "feature_artifact": _sha256(temporary / "feature_artifact.npz"),
                "hierarchy": _sha256(temporary / "meddra_hierarchy.json"),
                "manifest": _sha256(temporary / "benchmark" / "manifest.json"),
            },
        }
        if onnx_dir is not None:
            onnx_metadata = json.loads(
                (temporary / "onnx" / "onnx_release.json").read_text(encoding="utf-8")
            )
            expected_sources = onnx_metadata.get("source_sha256", {})
            if expected_sources.get("checkpoint") != metadata["sha256"]["checkpoint"]:
                raise ValueError("ONNX/checkpoint source hash mismatch")
            if expected_sources.get("selection") != metadata["sha256"]["selection"]:
                raise ValueError("ONNX/selection source hash mismatch")
            expected_onnx = onnx_metadata.get("onnx_sha256", {})
            actual_onnx = {
                "fused": _sha256(temporary / "onnx" / "teacher_fused.onnx"),
                "baseline": _sha256(temporary / "onnx" / "teacher_baseline.onnx"),
            }
            if expected_onnx != actual_onnx:
                raise ValueError("ONNX graph hash mismatch")
            metadata["runtimes"] = {
                "default": "onnx",
                "available": ["onnx", "pytorch"],
                "onnx": {
                    "fused": "onnx/teacher_fused.onnx",
                    "baseline": "onnx/teacher_baseline.onnx",
                    "metadata": "onnx/onnx_release.json",
                    "sha256": actual_onnx,
                },
            }
        (temporary / "release.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if output.exists():
            raise FileExistsError(f"output already exists: {output}")
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(f"MODEL_RELEASE_COMPLETE: {output}")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

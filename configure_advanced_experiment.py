"""Materialize hash-pinned teacher or student YAML from validated artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import yaml

from run_precomputed_experiment import _validate_teacher_selection, preflight_experiment
from src.data.manifest_dataset import load_manifest_records
from src.features.cached_token_artifact import CachedTokenArtifact
from src.features.multimodal_feature_artifact import MultimodalFeatureArtifact
from src.models.hierarchy import load_hierarchy_mapping
from src.models.factory import derive_run_model_id, validate_enabled_modalities
from src.training.engine import verify_manifest


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("teacher", "student"), required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--hierarchy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--teacher-config", type=Path)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--teacher-selection", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = yaml.safe_load(args.template.read_text())
    if not isinstance(payload, dict):
        raise ValueError("template must be a YAML mapping")
    manifest_path = args.manifest.resolve()
    manifest = verify_manifest(str(manifest_path))
    labels = load_manifest_records(manifest_path, manifest, "train").labels
    hierarchy = load_hierarchy_mapping(
        args.hierarchy, selected_specific_cuis=[str(label["cui"]) for label in labels]
    )
    payload["num_specific"] = len(labels)
    payload["num_organ"] = len(hierarchy.organ_order)
    payload["hierarchy_path"] = str(args.hierarchy.resolve())
    payload["hierarchy_sha256"] = _hash(args.hierarchy)
    labels_path = Path(manifest["labels_path"])
    if not labels_path.is_absolute():
        labels_path = manifest_path.parent / labels_path
    payload["manifest_hash"] = manifest["manifest_hash"]
    payload["labels_artifact_sha256"] = _hash(labels_path)
    payload["labels_order_sha256"] = hashlib.sha256(
        json.dumps(
            [str(label["cui"]) for label in labels],
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()

    if args.model == "teacher":
        if args.features is None:
            raise ValueError("teacher configuration requires --features")
        artifact = MultimodalFeatureArtifact.load(args.features)
        compatibility = artifact.metadata["manifest_compatibility"]
        for key in ("benchmark_id", "scenario", "seed", "manifest_hash"):
            if compatibility[key] != manifest[key]:
                raise ValueError(f"feature artifact is incompatible with manifest field {key}")
        payload["enabled_modalities"] = list(
            validate_enabled_modalities(payload.get("enabled_modalities"))
        )
        payload.update(
            {
                "model_type": "multimodal_teacher",
                "morgan_dim": artifact.morgan.shape[1],
                "molformer_dim": artifact.molformer_tokens.shape[2],
                "mpnn_dim": artifact.mpnn_tokens.shape[2],
                "kg_dim": artifact.kg_tokens.shape[2],
                "molformer_token_count": artifact.molformer_tokens.shape[1],
                "mpnn_token_count": artifact.mpnn_tokens.shape[1],
                "kg_token_count": artifact.kg_tokens.shape[1],
                "feature_artifact_path": str(args.features.resolve()),
                "feature_artifact_sha256": _hash(args.features),
            }
        )
    else:
        missing = [
            name
            for name, value in (
                ("--cache", args.cache),
                ("--teacher-config", args.teacher_config),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"student configuration requires {', '.join(missing)}")
        teacher_config_payload = yaml.safe_load(args.teacher_config.read_text())
        if not isinstance(teacher_config_payload, dict):
            raise ValueError("teacher config must be a YAML mapping")
        teacher_model_id = derive_run_model_id(teacher_config_payload)
        teacher_checkpoint = args.teacher_checkpoint or Path("artifacts/runs") / (
            f"{teacher_model_id}__{manifest['scenario']}__seed_{manifest['seed']}__"
            f"{manifest['manifest_hash']}"
        ) / "checkpoint_best.pt"
        teacher_selection = args.teacher_selection or teacher_checkpoint.with_name(
            "teacher_validation_selection.json"
        )
        selection = _validate_teacher_selection(teacher_selection)
        selection_hash = _hash(teacher_selection)
        cache = CachedTokenArtifact.load(args.cache)
        payload.update(
            {
                "model_type": "distilled_pair_student",
                "token_dim": cache.token_dim,
                "token_count": cache.token_count,
                "cached_token_artifact_path": str(args.cache.resolve()),
                "cached_token_artifact_sha256": _hash(args.cache),
                "teacher_config_path": str(args.teacher_config.resolve()),
                "teacher_config_sha256": _hash(args.teacher_config),
                "teacher_checkpoint_path": str(teacher_checkpoint.resolve()),
                "teacher_checkpoint_sha256": _hash(teacher_checkpoint),
                "teacher_selection_path": str(teacher_selection.resolve()),
                "teacher_selection_sha256": selection_hash,
                "teacher_selected_mode": selection["selected"]["mode"],
            }
        )
    _write_yaml(args.output, payload)
    preflight_experiment(args.output, manifest_path)
    print(f"Wrote and preflight-validated {args.model} config to {args.output}")


if __name__ == "__main__":
    main()

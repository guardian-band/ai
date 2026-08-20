"""Export the trained teacher's fixed latent tokens for CPU-student training."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from run_precomputed_experiment import (
    _derive_run_model_id,
    _validate_teacher_selection,
    preflight_experiment,
)
from src.features.cached_token_artifact import build_cached_token_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--selection-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    return parser.parse_args()


def _resolve_teacher_selection(
    path: Path, *, expected_hash: str | None = None
) -> tuple[dict, str]:
    """Validate the selection file and return its payload plus actual digest."""

    selection = _validate_teacher_selection(path, expected_hash=expected_hash)
    selection_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    return selection, selection_hash


def _batches(artifact, batch_size):
    names = (
        "morgan",
        "molformer_tokens",
        "mpnn_tokens",
        "kg_tokens",
        "morgan_available",
        "molformer_available",
        "mpnn_available",
        "kg_available",
        "molformer_padding_mask",
        "mpnn_padding_mask",
        "kg_padding_mask",
    )
    for start in range(0, len(artifact.drug_ids), batch_size):
        stop = min(start + batch_size, len(artifact.drug_ids))
        yield {
            "drug_ids": artifact.drug_ids[start:stop],
            "inputs": {
                name: torch.from_numpy(np.array(getattr(artifact, name)[start:stop], copy=True))
                for name in names
            },
        }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    plan = preflight_experiment(args.experiment, args.benchmark)
    if plan.config["model_type"] != "multimodal_teacher" or plan.feature_artifact is None:
        raise ValueError("experiment must declare a validated multimodal_teacher")
    checkpoint = args.checkpoint or Path("artifacts/runs") / (
        f"{_derive_run_model_id(plan.config)}__{plan.manifest['scenario']}__seed_{plan.manifest['seed']}__"
        f"{plan.manifest['manifest_hash']}"
    ) / "checkpoint_best.pt"
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if args.checkpoint_sha256 is not None and checkpoint_hash != args.checkpoint_sha256:
        raise ValueError("teacher checkpoint hash mismatch")
    selection_path = args.selection or checkpoint.with_name("teacher_validation_selection.json")
    selection, selection_hash = _resolve_teacher_selection(
        selection_path, expected_hash=args.selection_sha256
    )
    selected_mode = selection["selected"]["mode"]
    model = plan.model.to(args.device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=args.device, weights_only=True), strict=True
    )
    model.set_training_stage(selected_mode)
    model.eval()
    feature_path = Path(plan.config["feature_artifact_path"])
    if not feature_path.is_absolute():
        feature_path = plan.config_path.parent / feature_path
    feature_hash = hashlib.sha256(feature_path.read_bytes()).hexdigest()
    artifact = build_cached_token_artifact(
        model,
        _batches(plan.feature_artifact, args.batch_size),
        args.output,
        teacher_checkpoint_hash=checkpoint_hash,
        teacher_config_hash=hashlib.sha256(args.experiment.read_bytes()).hexdigest(),
        teacher_selection_hash=selection_hash,
        teacher_selected_mode=selected_mode,
        multimodal_feature_artifact_hash=feature_hash,
    )
    print(f"Wrote {len(artifact.drug_ids)} teacher cache rows to {args.output}")


if __name__ == "__main__":
    main()

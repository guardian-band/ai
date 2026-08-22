#!/usr/bin/env python3
"""Evaluate the frozen warm Morgan model's 15 MedDRA organ-system outputs.

This command never trains or selects a model. Validation predictions are used
to fit temperature scaling and thresholds. Only after those values are frozen
is the test split evaluated once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from run_precomputed_experiment import (
    _required_path,
    _teacher_or_student_batch,
    preflight_experiment,
)
from src.data.precomputed_datasets import MultimodalPairDataset
from src.evaluation.calibration import apply_temperature, calibrate_validation_logits
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.thresholds import select_thresholds


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(_json_value(payload), stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _predict_organ(
    model: torch.nn.Module,
    dataset: MultimodalPairDataset,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size):
            drug_a, drug_b, organ, *_ = _teacher_or_student_batch(batch, device, False)
            output = model(drug_a, drug_b)
            logits.append(output.organ_logits.detach().cpu().numpy())
            targets.append(organ.detach().cpu().numpy())
    return np.concatenate(logits), np.concatenate(targets)


def _prediction_frame(
    dataset: MultimodalPairDataset,
    names: list[str],
    logits: np.ndarray,
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    if logits.shape != targets.shape or probabilities.shape != targets.shape:
        raise ValueError("organ logits, targets, and probabilities must have equal shapes")
    if logits.shape[1] != len(names):
        raise ValueError("organ prediction columns do not match hierarchy order")
    frame = pd.DataFrame({"pair_id": [str(item["pair_id"]) for item in dataset.records]})
    for index, name in enumerate(names):
        frame[f"truth__{name}"] = targets[:, index]
        frame[f"prob__{name}"] = probabilities[:, index]
        frame[f"logit__{name}"] = logits[:, index]
    return frame


def _organ_prevalences(
    dataset: MultimodalPairDataset, organ_names: list[str]
) -> dict[str, float]:
    targets = np.zeros((len(dataset.records), len(organ_names)), dtype=np.float32)
    for row_index, record in enumerate(dataset.records):
        for label_index, active in enumerate(record["labels"]):
            if float(active) <= 0:
                continue
            cui = str(dataset.labels[label_index]["cui"])
            targets[row_index, dataset.hierarchy.specific_to_organ[cui]] = 1.0
    return {
        name: float(targets[:, organ_index].mean())
        for organ_index, name in enumerate(organ_names)
    }


def evaluate_level1(
    experiment: Path,
    manifest: Path,
    checkpoint: Path,
    output: Path,
) -> dict[str, Any]:
    plan = preflight_experiment(experiment, manifest)
    if plan.config.get("model_type") != "multimodal_teacher":
        raise ValueError("Level-1 evaluation requires a Teacher experiment config")
    if plan.feature_artifact is None:
        raise ValueError("preflight did not load the multimodal feature artifact")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"frozen Morgan checkpoint is missing: {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = plan.model.to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if not isinstance(state, Mapping):
        raise ValueError("frozen Morgan checkpoint must contain a state dict")
    model.load_state_dict(state)
    model.set_training_stage("baseline")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    organ_names = [str(item) for item in plan.hierarchy.organ_order]
    if len(organ_names) != 15:
        raise ValueError(f"expected 15 Level-1 organ systems, found {len(organ_names)}")
    batch_size = int(plan.config.get("batch_size", 256))

    print("[level1-validation] generating organ logits", flush=True)
    validation_logits, validation_targets = _predict_organ(
        model, plan.validation_dataset, device, batch_size
    )
    validation_logits_frame = pd.DataFrame(validation_logits, columns=organ_names)
    validation_targets_frame = pd.DataFrame(validation_targets, columns=organ_names)
    calibrated, temperatures = calibrate_validation_logits(
        {"organ": validation_logits_frame}, {"organ": validation_targets_frame}
    )
    validation_probabilities = calibrated["organ"].to_numpy()
    thresholds = select_thresholds(
        validation_targets_frame, calibrated["organ"]
    )

    hierarchy_path = _required_path(plan.config_path, plan.config, "hierarchy_path")
    test_dataset = MultimodalPairDataset.from_manifest(
        plan.manifest_path,
        plan.manifest,
        "test",
        plan.feature_artifact,
        hierarchy_path,
    )
    print("[level1-test] calibration frozen; generating final organ predictions", flush=True)
    test_logits, test_targets = _predict_organ(model, test_dataset, device, batch_size)
    test_probabilities = apply_temperature(test_logits, temperatures["organ"])
    train_prevalences = _organ_prevalences(plan.train_dataset, organ_names)
    metrics = compute_all_metrics(
        pd.DataFrame(test_targets, columns=organ_names),
        pd.DataFrame(test_probabilities, columns=organ_names),
        thresholds,
        train_prevalences,
    )

    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        output / "level1_calibration.json",
        {"temperature": temperatures["organ"], "fit_split": "validation"},
    )
    _atomic_json(output / "level1_thresholds.json", thresholds)
    _atomic_json(output / "level1_test_metrics.json", metrics)
    _prediction_frame(
        plan.validation_dataset,
        organ_names,
        validation_logits,
        validation_targets,
        validation_probabilities,
    ).to_parquet(output / "level1_validation_predictions.parquet", index=False)
    _prediction_frame(
        test_dataset,
        organ_names,
        test_logits,
        test_targets,
        test_probabilities,
    ).to_parquet(output / "level1_test_predictions.parquet", index=False)
    rows = [dict(organ=name, **values) for name, values in metrics["per_label_metrics"].items()]
    pd.DataFrame(rows).to_csv(output / "level1_per_organ_metrics.csv", index=False)

    report = {
        "schema_version": 1,
        "status": "complete",
        "level": 1,
        "target": "15 MedDRA organ systems",
        "model_family": "Morgan-only pair predictor",
        "scenario": str(plan.manifest["scenario"]),
        "seed": int(plan.manifest["seed"]),
        "selection_split": "validation",
        "test_metrics_used_for_selection": False,
        "evaluation_device": str(device),
        "checkpoint_sha256": _sha256(checkpoint),
        "experiment_sha256": _sha256(experiment),
        "manifest_sha256": _sha256(manifest),
        "organ_order": organ_names,
        "test_metrics": metrics,
    }
    _atomic_json(output / "level1_evaluation_manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_level1(
        args.experiment.resolve(),
        args.manifest.resolve(),
        args.checkpoint.resolve(),
        args.output.resolve(),
    )
    metrics = report["test_metrics"]
    print(
        "LEVEL1_COMPLETE "
        f"macro_auprc={metrics['macro_ap']:.6f} "
        f"micro_auprc={metrics['micro_ap']:.6f} "
        f"macro_auroc={metrics['macro_auroc']:.6f} "
        f"micro_auroc={metrics['micro_auroc']:.6f} "
        f"precision_at_5={metrics['precision_at_k']:.6f} "
        f"recall_at_5={metrics['recall_at_k']:.6f} "
        f"ndcg_at_5={metrics['ndcg_at_k']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

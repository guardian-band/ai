#!/usr/bin/env python3
"""Calibrate, evaluate, and CPU-benchmark an already frozen Morgan baseline.

This command never trains or selects a model.  It fits calibration and label
thresholds on validation predictions, opens the test split only after those
artifacts are frozen, and benchmarks one CPU pair through the exact checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from run_experiment import _persist_per_label_metrics, _persist_prediction_artifacts
from run_precomputed_experiment import (
    _predict,
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


def _labels(dataset: MultimodalPairDataset) -> list[str]:
    labels = [str(item["cui"]) for item in dataset.labels]
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("manifest labels must be unique")
    return labels


def _benchmark_pair(
    model: torch.nn.Module,
    dataset: MultimodalPairDataset,
    *,
    warmups: int,
    iterations: int,
    threads: int,
) -> dict[str, Any]:
    if warmups < 1 or iterations < 1 or threads < 1:
        raise ValueError("warmups, iterations, and threads must be positive")
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    drug_a, drug_b, *_ = _teacher_or_student_batch(batch, torch.device("cpu"), False)
    original_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        with torch.inference_mode():
            forward = model(drug_a, drug_b).specific_logits
            reverse = model(drug_b, drug_a).specific_logits
            max_order_delta = float(torch.max(torch.abs(forward - reverse)).item())
            if max_order_delta > 1e-6:
                raise RuntimeError("Morgan pair prediction is not order invariant")
            for _ in range(warmups):
                model(drug_a, drug_b)
            timings_ms: list[float] = []
            for _ in range(iterations):
                started = time.perf_counter_ns()
                model(drug_a, drug_b)
                timings_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    finally:
        torch.set_num_threads(original_threads)
    values = np.asarray(timings_ms, dtype=float)
    record = dataset.records[0]
    return {
        "drug_a": str(record["drug_a"]),
        "drug_b": str(record["drug_b"]),
        "pair_id": str(record["pair_id"]),
        "device": "cpu",
        "torch_threads": threads,
        "warmups": warmups,
        "iterations": iterations,
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(values.max()),
        "order_invariance_max_abs_logit_delta": max_order_delta,
    }


def finalize_release(
    experiment: Path,
    manifest: Path,
    checkpoint: Path,
    output: Path,
    *,
    warmups: int = 50,
    iterations: int = 500,
    threads: int = 1,
) -> dict[str, Any]:
    plan = preflight_experiment(experiment, manifest)
    if plan.config.get("model_type") != "multimodal_teacher":
        raise ValueError("frozen Morgan evaluation requires a Teacher experiment config")
    if plan.feature_artifact is None:
        raise ValueError("preflight did not load the multimodal feature artifact")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"frozen Morgan checkpoint is missing: {checkpoint}")

    evaluation_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = plan.model.to(evaluation_device)
    state = torch.load(checkpoint, map_location=evaluation_device, weights_only=True)
    if not isinstance(state, Mapping):
        raise ValueError("frozen Morgan checkpoint must contain a state dict")
    model.load_state_dict(state)
    model.set_training_stage("baseline")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    batch_size = int(plan.config.get("batch_size", 256))
    validation_loader = DataLoader(plan.validation_dataset, batch_size=batch_size)
    print("[validation] generating frozen Morgan logits", flush=True)
    validation_logits, validation_targets = _predict(
        model, validation_loader, evaluation_device, False
    )
    labels = _labels(plan.validation_dataset)
    validation_logits_frame = pd.DataFrame(validation_logits, columns=labels)
    validation_targets_frame = pd.DataFrame(validation_targets, columns=labels)
    calibrated, temperatures = calibrate_validation_logits(
        {"specific": validation_logits_frame},
        {"specific": validation_targets_frame},
    )
    thresholds = select_thresholds(validation_targets_frame, calibrated["specific"])

    # Only now, after validation-only calibration and thresholds are fixed, is
    # the test dataset constructed and evaluated exactly once.
    hierarchy_path = _required_path(plan.config_path, plan.config, "hierarchy_path")
    test_dataset = MultimodalPairDataset.from_manifest(
        plan.manifest_path,
        plan.manifest,
        "test",
        plan.feature_artifact,
        hierarchy_path,
    )
    print("[test] calibration and thresholds frozen; generating final predictions", flush=True)
    test_logits, test_targets = _predict(
        model,
        DataLoader(test_dataset, batch_size=batch_size),
        evaluation_device,
        False,
    )
    test_probabilities = apply_temperature(test_logits, temperatures["specific"])
    train_prevalences = {
        label: float(np.mean([record["labels"][index] for record in plan.train_dataset.records]))
        for index, label in enumerate(labels)
    }
    metrics = compute_all_metrics(
        pd.DataFrame(test_targets, columns=labels),
        pd.DataFrame(test_probabilities, columns=labels),
        thresholds,
        train_prevalences,
    )
    # Final batch evaluation may use CUDA for speed, but the deployment timing
    # below always moves the exact frozen model to CPU.
    model = model.to("cpu")
    latency = _benchmark_pair(
        model,
        plan.validation_dataset,
        warmups=warmups,
        iterations=iterations,
        threads=threads,
    )

    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        output / "calibration.json",
        {"temperatures": temperatures, "fit_split": "validation", "input": "raw_logits"},
    )
    _atomic_json(output / "thresholds.json", thresholds)
    _atomic_json(output / "final_test_metrics.json", metrics)
    _atomic_json(output / "cpu_pair_benchmark.json", latency)
    _persist_prediction_artifacts(
        str(output),
        "validation",
        plan.validation_dataset,
        validation_logits,
        validation_targets,
        calibrated["specific"].to_numpy(),
    )
    _persist_prediction_artifacts(
        str(output),
        "test",
        test_dataset,
        test_logits,
        test_targets,
        test_probabilities,
    )
    _persist_per_label_metrics(str(output), metrics)

    report = {
        "schema_version": 1,
        "status": "complete",
        "model_family": "Morgan-only pair predictor",
        "scenario": str(plan.manifest["scenario"]),
        "seed": int(plan.manifest["seed"]),
        "selection_split": "validation",
        "test_metrics_used_for_selection": False,
        "batch_evaluation_device": str(evaluation_device),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "experiment_sha256": _sha256(experiment),
        "manifest_sha256": _sha256(manifest),
        "calibration_temperature": float(temperatures["specific"]),
        "test_metrics": metrics,
        "cpu_single_pair": latency,
        "artifacts": {
            name: {"sha256": _sha256(output / name), "bytes": (output / name).stat().st_size}
            for name in (
                "calibration.json",
                "thresholds.json",
                "final_test_metrics.json",
                "cpu_pair_benchmark.json",
                "per_label_metrics.csv",
            )
        },
    }
    _atomic_json(output / "release_evaluation_manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    report = finalize_release(
        args.experiment.resolve(),
        args.manifest.resolve(),
        args.checkpoint.resolve(),
        args.output.resolve(),
        warmups=args.warmups,
        iterations=args.iterations,
        threads=args.threads,
    )
    metrics = report["test_metrics"]
    latency = report["cpu_single_pair"]
    print(
        "FINAL_MORGAN_RELEASE "
        f"macro_auprc={metrics['macro_ap']:.6f} micro_auprc={metrics['micro_ap']:.6f} "
        f"precision_at_5={metrics['precision_at_k']:.6f} "
        f"recall_at_5={metrics['recall_at_k']:.6f} ndcg_at_5={metrics['ndcg_at_k']:.6f} "
        f"cpu_p50_ms={latency['p50_ms']:.3f} cpu_p95_ms={latency['p95_ms']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

"""Audit an existing cold Morgan+MPNN run without retraining or touching test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_precomputed_experiment import (  # noqa: E402
    _predict_teacher_variants,
    preflight_experiment,
)


def _macro_ap(logits: np.ndarray, targets: np.ndarray) -> float:
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    values = [
        average_precision_score(targets[:, i], probabilities[:, i])
        for i in range(targets.shape[1])
        if np.unique(targets[:, i]).size == 2
    ]
    return float(np.mean(values)) if values else float("nan")


def _feature_stats(artifact, drug_ids: set[str]) -> dict[str, float | int]:
    indices = [i for i, drug_id in enumerate(artifact.drug_ids) if drug_id in drug_ids]
    available = np.asarray(artifact.mpnn_available[indices], dtype=bool)
    selected = np.asarray(indices, dtype=int)[available]
    tokens = np.asarray(artifact.mpnn_tokens[selected], dtype=np.float64)
    masks = np.asarray(artifact.mpnn_padding_mask[selected], dtype=bool)
    valid = (~masks)[..., None]
    pooled = (tokens * valid).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
    norms = np.linalg.norm(pooled, axis=1)
    return {
        "drug_count": len(indices),
        "available_count": int(available.sum()),
        "coverage": float(available.mean()) if len(available) else float("nan"),
        "feature_mean": float(pooled.mean()),
        "feature_std": float(pooled.std()),
        "mean_drug_norm": float(norms.mean()),
        "std_drug_norm": float(norms.std()),
        "mean_dimension_variance": float(pooled.var(axis=0).mean()),
    }


def _bootstrap_delta(
    baseline: np.ndarray,
    fused: np.ndarray,
    targets: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(samples):
        indices = rng.integers(0, len(targets), len(targets))
        delta = (
            _macro_ap(fused[indices], targets[indices])
            - _macro_ap(baseline[indices], targets[indices])
        )
        if np.isfinite(delta):
            deltas.append(delta)
    values = np.asarray(deltas)
    return {
        "samples_requested": samples,
        "samples_valid": int(values.size),
        "mean": float(values.mean()),
        "ci_2_5": float(np.quantile(values, 0.025)),
        "ci_97_5": float(np.quantile(values, 0.975)),
        "probability_positive": float((values > 0).mean()),
        "probability_above_promotion_delta": float((values > 0.002).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    args = parser.parse_args()

    plan = preflight_experiment(args.experiment, args.benchmark)
    if tuple(plan.model.enabled_modalities) != ("mpnn",):
        raise ValueError("audit requires a Morgan+MPNN teacher configuration")
    fused_checkpoint = args.run_dir / "checkpoint_fused.pt"
    selection_path = args.run_dir / "teacher_validation_selection.json"
    model = plan.model.to("cpu")
    model.load_state_dict(torch.load(fused_checkpoint, map_location="cpu", weights_only=True))
    model.set_training_stage("fused")
    loader = DataLoader(plan.validation_dataset, batch_size=int(plan.config["batch_size"]))
    predictions, diagnostics = _predict_teacher_variants(model, loader, torch.device("cpu"))
    baseline_logits, targets = predictions["baseline"]
    fused_logits, _ = predictions["combined"]
    gated_correction = fused_logits - baseline_logits

    baseline_probs = 1.0 / (1.0 + np.exp(-baseline_logits))
    fused_probs = 1.0 / (1.0 + np.exp(-fused_logits))
    label_names = [str(item["cui"]) for item in plan.validation_dataset.labels]
    per_label = []
    for index, label in enumerate(label_names):
        if np.unique(targets[:, index]).size != 2:
            continue
        base_ap = average_precision_score(targets[:, index], baseline_probs[:, index])
        fused_ap = average_precision_score(targets[:, index], fused_probs[:, index])
        per_label.append({"label": label, "baseline_ap": float(base_ap), "fused_ap": float(fused_ap), "delta": float(fused_ap - base_ap)})
    per_label.sort(key=lambda row: row["delta"], reverse=True)

    train_drugs = {
        str(record[key])
        for record in plan.train_dataset.records
        for key in ("drug_a", "drug_b")
    }
    validation_drugs = {
        str(record[key])
        for record in plan.validation_dataset.records
        for key in ("drug_a", "drug_b")
    }
    cold_drugs = validation_drugs - train_drugs
    selection = json.loads(selection_path.read_text())
    gate_values = diagnostics.get("gates", {}).get("mpnn", [])
    correction_norm = np.linalg.norm(gated_correction, axis=1)
    baseline_norm = np.linalg.norm(baseline_logits, axis=1)
    report = {
        "scope": "validation_only_no_retraining",
        "gate_history_available": False,
        "gate_history_note": "Existing run stores only the validation-selected checkpoint, not per-epoch gate history.",
        "final_mpnn_gates": gate_values,
        "selection": selection,
        "validation_macro_ap": {
            "baseline": _macro_ap(baseline_logits, targets),
            "fused": _macro_ap(fused_logits, targets),
            "delta": _macro_ap(fused_logits, targets) - _macro_ap(baseline_logits, targets),
        },
        "gated_correction_vs_baseline_logits": {
            "mean_abs_correction": float(np.abs(gated_correction).mean()),
            "mean_pair_correction_norm": float(correction_norm.mean()),
            "mean_pair_baseline_norm": float(baseline_norm.mean()),
            "mean_norm_ratio": float((correction_norm / np.maximum(baseline_norm, 1e-12)).mean()),
            "mean_abs_probability_change": float(np.abs(fused_probs - baseline_probs).mean()),
            "max_abs_probability_change": float(np.abs(fused_probs - baseline_probs).max()),
        },
        "mpnn_feature_stats": {
            "training_drugs": _feature_stats(plan.feature_artifact, train_drugs),
            "cold_validation_drugs": _feature_stats(plan.feature_artifact, cold_drugs),
        },
        "bootstrap_validation_delta": _bootstrap_delta(
            baseline_logits, fused_logits, targets,
            samples=args.bootstrap_samples,
            seed=int(plan.manifest["seed"]),
        ),
        "per_label": {
            "improved_count": sum(row["delta"] > 0 for row in per_label),
            "degraded_count": sum(row["delta"] < 0 for row in per_label),
            "unchanged_count": sum(row["delta"] == 0 for row in per_label),
            "top_improvements": per_label[:10],
            "top_degradations": list(reversed(per_label[-10:])),
        },
        "provenance": dict(plan.feature_artifact.metadata),
        "leakage_note": "Artifact provenance is reported, but absence of held-out-label leakage cannot be proven from embeddings alone.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"AUDIT_COMPLETE: {args.output}", flush=True)


if __name__ == "__main__":
    main()

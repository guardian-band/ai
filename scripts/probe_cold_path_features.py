#!/usr/bin/env python3
"""Test cheap directed path features as a correction to a frozen Morgan model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_precomputed_experiment import _predict, preflight_experiment  # noqa: E402
from src.features.directed_path_features import (  # noqa: E402
    DirectedPathFeatureIndex,
    degree_matched_permutation,
)


def macro_ap(logits: np.ndarray, targets: np.ndarray) -> float:
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
    values = [
        average_precision_score(targets[:, i], probabilities[:, i])
        for i in range(targets.shape[1])
        if np.unique(targets[:, i]).size == 2
    ]
    return float(np.mean(values))


class CorrectionMLP(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(inputs, 32), nn.ReLU(), nn.Linear(32, outputs))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, values):
        return self.network(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--primekg", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=500)
    args = parser.parse_args()

    plan = preflight_experiment(args.experiment, args.manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = plan.model.to(device)
    model.load_state_dict(torch.load(args.baseline_checkpoint, map_location=device, weights_only=True))
    model.set_training_stage("baseline")
    train_loader = DataLoader(plan.train_dataset, batch_size=int(plan.config["batch_size"]))
    validation_loader = DataLoader(plan.validation_dataset, batch_size=int(plan.config["batch_size"]))
    train_logits, train_targets = _predict(model, train_loader, device, False)
    validation_logits, validation_targets = _predict(model, validation_loader, device, False)

    primekg = pd.read_parquet(args.primekg) if args.primekg.suffix == ".parquet" else pd.read_csv(args.primekg)
    labels = [str(item["cui"]) for item in plan.validation_dataset.labels]
    index = DirectedPathFeatureIndex(primekg, labels)
    train_records = [dict(row) for row in plan.train_dataset.records]
    validation_records = [dict(row) for row in plan.validation_dataset.records]
    train_features = index.transform_records(train_records)
    validation_features = index.transform_records(validation_records)
    mean = train_features.mean(axis=0)
    std = np.maximum(train_features.std(axis=0), 1e-6)
    train_features = (train_features - mean) / std
    validation_features = (validation_features - mean) / std

    seed = int(plan.manifest["seed"])
    torch.manual_seed(seed)
    correction = CorrectionMLP(train_features.shape[1], train_targets.shape[1]).to(device)
    optimizer = torch.optim.AdamW(correction.parameters(), lr=1e-3, weight_decay=1e-4)
    positives = train_targets.sum(axis=0)
    pos_weight = np.clip((len(train_targets) - positives) / np.maximum(positives, 1), 1, 50)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device, dtype=torch.float32))
    dataset = TensorDataset(
        torch.tensor(train_features), torch.tensor(train_logits), torch.tensor(train_targets)
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=512, shuffle=True, generator=generator)
    val_x = torch.tensor(validation_features, device=device)
    best_ap = macro_ap(validation_logits, validation_targets)
    baseline_ap = best_ap
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        correction.train()
        losses = []
        for batch_x, batch_logits, batch_targets in loader:
            batch_x = batch_x.to(device)
            batch_logits = batch_logits.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(batch_logits + correction(batch_x), batch_targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        correction.eval()
        with torch.inference_mode():
            fused = validation_logits + correction(val_x).cpu().numpy()
        score = macro_ap(fused, validation_targets)
        improved = score > best_ap
        if improved:
            best_ap = score
            best_state = {key: value.detach().cpu().clone() for key, value in correction.state_dict().items()}
            stale = 0
        else:
            stale += 1
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_macro_auprc": score})
        print(f"[epoch {epoch:02d}] loss={np.mean(losses):.6f} val_macro_auprc={score:.6f}", flush=True)
        if stale >= args.patience:
            break

    pair_positive = np.asarray(
        [str(row.get("observation_status", "")).lower() in {"observed_positive", "positive", "known_positive"} for row in validation_records]
    )
    raw_validation_features = index.transform_records(validation_records)
    permutation = degree_matched_permutation(
        raw_validation_features, pair_positive, samples=args.permutations, seed=seed
    )
    permutation["feature_names"] = list(index.feature_names)
    delta = best_ap - baseline_ap
    if delta >= 0.002:
        decision = "develop_pair_conditioned_teacher"
    elif any(
        p < 0.05 and difference > 0
        for p, difference in zip(permutation["two_sided_p_value"], permutation["observed_mean_difference"])
    ):
        decision = "consider_one_small_emergnn_prototype"
    else:
        decision = "stop_expensive_graph_model"
    report = {
        "scope": "train_and_validation_only_no_test_access",
        "graph_semantics": "directed_with_explicit_inverse_relation_identity",
        "feature_names": list(index.feature_names),
        "baseline_validation_macro_auprc": baseline_ap,
        "best_correction_validation_macro_auprc": best_ap,
        "delta": delta,
        "promotion_threshold": 0.002,
        "decision": decision,
        "history": history,
        "degree_matched_positive_control_permutation": permutation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    if best_state is not None:
        torch.save(best_state, args.output.with_suffix(".pt"))
    print(json.dumps(report, indent=2), flush=True)
    print(f"PATH_PROBE_COMPLETE: {args.output}", flush=True)


if __name__ == "__main__":
    main()

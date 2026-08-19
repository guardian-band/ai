import argparse
import os
import json
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from src.training.engine import REQUIRED_RUN_ARTIFACTS, StateGuardedTrainer, verify_manifest
from src.training.reproducibility import set_global_seed, collect_environment_info
from src.data.manifest_dataset import ManifestPolypharmacyDataset
from src.models.factory import (
    PrevalenceModel,
    create_model,
    load_experiment_config,
    validate_model_type,
    validate_precomputed_model_runner,
)
from src.evaluation.calibration import apply_temperature, calibrate_validation_logits
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.thresholds import select_thresholds


def _label_names(dataset):
    """Return the ordered CUI labels declared by a manifest dataset."""

    labels = getattr(dataset, "labels", None)
    if not labels:
        raise ValueError("manifest dataset must expose a non-empty labels definition")
    names = [str(record["cui"]) for record in labels]
    if len(names) != len(set(names)):
        raise ValueError("manifest dataset labels must have unique cui values")
    return names


def _pair_ids(dataset, count):
    records = getattr(dataset, "records", None)
    if records is None:
        return [str(index) for index in range(count)]
    if len(records) != count:
        raise ValueError("prediction dataset records do not match prediction count")
    # Fixture datasets used by existing callers predate pair_id.  Preserve
    # their entry point with deterministic positional IDs while real manifest
    # datasets always provide the artifact pair_id.
    return [str(record.get("pair_id", index)) for index, record in enumerate(records)]


def _train_label_matrix(dataset, num_labels):
    records = getattr(dataset, "records", None)
    if records is not None:
        matrix = np.asarray([record["labels"] for record in records], dtype=float)
    else:
        matrix = np.asarray(
            [dataset[index][4].detach().cpu().numpy() for index in range(len(dataset))],
            dtype=float,
        )
    if matrix.ndim != 2 or matrix.shape[1] != num_labels:
        raise ValueError("train dataset labels do not match the manifest label definition")
    return matrix


def _prediction_frames(dataset, logits, targets, probabilities):
    """Build the run artifact tables while preserving dataset record order."""

    label_names = _label_names(dataset)
    logits = np.asarray(logits)
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)
    if logits.shape != targets.shape or probabilities.shape != targets.shape:
        raise ValueError("logits, probabilities, and targets must have identical shapes")
    if logits.ndim != 2 or logits.shape[1] != len(label_names):
        raise ValueError("prediction columns do not match manifest labels")
    pair_ids = _pair_ids(dataset, targets.shape[0])
    truth = pd.DataFrame(targets, columns=[f"truth__{label}" for label in label_names])
    probs = pd.DataFrame(probabilities, columns=[f"prob__{label}" for label in label_names])
    logits_frame = pd.DataFrame(logits, columns=[f"logit__{label}" for label in label_names])
    predictions = pd.concat([truth, probs], axis=1)
    predictions.insert(0, "pair_id", pair_ids)
    logits_frame.insert(0, "pair_id", pair_ids)
    return predictions, logits_frame


def _persist_prediction_artifacts(run_dir, split, dataset, logits, targets, probabilities):
    predictions, logits_frame = _prediction_frames(dataset, logits, targets, probabilities)
    predictions.to_parquet(os.path.join(run_dir, f"{split}_predictions.parquet"), index=False)
    logits_frame.to_parquet(os.path.join(run_dir, f"{split}_logits.parquet"), index=False)


def _persist_per_label_metrics(run_dir, metrics):
    per_label = metrics.get("per_label_metrics", {})
    rows = []
    for label_cui, values in per_label.items():
        row = {"label_cui": str(label_cui)}
        row.update(values)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(run_dir, "per_label_metrics.csv"), index=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, help="Path to experiment config YAML")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--benchmark", help="Path to benchmark manifest")
    group.add_argument("--all-benchmarks", action="store_true", help="Run over all discovered benchmark manifests")
    parser.add_argument("--benchmarks-root", default="artifacts/benchmarks", help="Root recursively searched by --all-benchmarks")
    args = parser.parse_args()
    return args


def discover_benchmark_manifests(benchmarks_root: str | Path, experiment_config_path: str | Path) -> list[Path]:
    root = Path(benchmarks_root)
    manifests = sorted(root.rglob("manifest.json")) if root.is_dir() else []
    if not manifests:
        raise ValueError(f"no benchmark manifests found under {root}")
    experiment_config = load_experiment_config(experiment_config_path)
    model_type = validate_model_type(experiment_config)
    verified: list[tuple[Path, dict]] = []
    seen: set[tuple[str, str, int, str]] = set()
    for path in manifests:
        try:
            manifest = verify_manifest(str(path))
            scenario = manifest.get("scenario")
            seed = manifest.get("seed")
            manifest_hash = manifest.get("manifest_hash")
            if not isinstance(manifest.get("benchmark_id"), str) or not manifest["benchmark_id"]:
                raise ValueError("missing benchmark_id")
            if not isinstance(scenario, str) or not scenario or isinstance(seed, bool) or not isinstance(seed, int) or not isinstance(manifest_hash, str):
                raise ValueError("missing scenario, seed, or manifest_hash")
            for artifact_key in ("pairs_path", "triples_path", "labels_path"):
                artifact = manifest.get(artifact_key)
                if not isinstance(artifact, str) or not (path.parent / artifact).is_file():
                    raise ValueError(f"missing {artifact_key} artifact")
            identity = (model_type, scenario, int(seed), manifest_hash)
            if identity in seen:
                raise ValueError(f"duplicate benchmark identity {identity}")
            seen.add(identity)
            verified.append((path, manifest))
        except Exception as exc:
            raise ValueError(f"invalid benchmark manifest {path}: {exc}") from exc
    return [path for path, _manifest in verified]

def run_single_experiment(experiment_config_path: str, manifest_path: str):
    if not manifest_path:
        raise ValueError("A benchmark manifest path is required for a single run.")

    # Verify the manifest before any run directory, seed, or training state is created.
    manifest = verify_manifest(manifest_path)
    experiment_config = load_experiment_config(experiment_config_path)
    validate_model_type(experiment_config)
    validate_precomputed_model_runner(experiment_config)
    
    seed = manifest["seed"]
    benchmark_id = manifest["benchmark_id"]
    benchmark_hash = manifest["manifest_hash"]
    scenario = manifest.get("scenario", "warm_pair")
    
    set_global_seed(seed)

    model_type = experiment_config["model_type"]
    run_id = f"{model_type}__{scenario}__seed_{seed}__{benchmark_hash}"
    
    run_dir = os.path.join("artifacts", "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    if os.path.exists(os.path.join(run_dir, "completion.json")):
        print(f"Skipping run {run_id} as it is already completed.")
        return
    
    with open(os.path.join(run_dir, "environment.json"), "w") as f:
        json.dump(collect_environment_info(), f, indent=2)
        
    resolved_config = dict(experiment_config)
    resolved_config.update({
        "experiment": experiment_config_path,
        "benchmark_id": benchmark_id,
        "scenario": scenario,
        "seed": seed,
        "manifest_hash": benchmark_hash,
    })
    with open(os.path.join(run_dir, "config.resolved.json"), "w") as f:
        json.dump(resolved_config, f, indent=2)
        
    trainer = StateGuardedTrainer(run_dir)

    drug_features_path = experiment_config.get("drug_features_path", "artifacts/morgan_fingerprints.parquet")
    train_dataset = ManifestPolypharmacyDataset.from_manifest(
        manifest_path, manifest, split="train", drug_features_path=drug_features_path
    )
    val_dataset = ManifestPolypharmacyDataset.from_manifest(
        manifest_path, manifest, split="validation", drug_features_path=drug_features_path
    )
    num_labels = len(train_dataset.labels)
    if num_labels == 0 or num_labels != len(val_dataset.labels):
        raise ValueError("train and validation label definitions must be non-empty and identical")
    label_names = _label_names(train_dataset)
    train_label_matrix = _train_label_matrix(train_dataset, num_labels)
    train_prevalences = {
        label: float(np.mean(train_label_matrix[:, index]))
        for index, label in enumerate(label_names)
    }
    resolved_config.update({
        "run_id": run_id,
        "model_type": experiment_config["model_type"],
        "train_prevalences": train_prevalences,
    })
    with open(os.path.join(run_dir, "config.resolved.json"), "w") as f:
        json.dump(resolved_config, f, indent=2)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    input_dim = experiment_config.get("input_dim", 2048)
    model = create_model(experiment_config, num_labels=num_labels, input_dim=input_dim).to(device)
    trainer.begin_training()
    print(f"Began training run {run_id}")

    batch_size = experiment_config.get("batch_size", 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    def predict(loader):
        """Return raw model logits and targets; sigmoid is applied by the caller."""
        model.eval()
        raw_logits, targets = [], []
        with torch.no_grad():
            for a_tok, b_tok, ma, mb, labels, _ in loader:
                a_tok, b_tok = a_tok.to(device), b_tok.to(device)
                ma, mb = ma.to(device), mb.to(device)
                logits = model(a_tok, b_tok, ma, mb)
                raw_logits.append(logits.cpu().numpy())
                targets.append(labels.numpy())
        return np.concatenate(raw_logits, axis=0), np.concatenate(targets, axis=0)

    best_val_ap = -1.0
    patience_counter = 0
    max_epochs = experiment_config.get("epochs", 100)

    if isinstance(model, PrevalenceModel):
        train_labels = torch.tensor(train_label_matrix, dtype=torch.float32)
        model.fit(train_labels)
        torch.save(model.state_dict(), os.path.join(run_dir, "checkpoint_best.pt"))
        val_logits, val_targets = predict(val_loader)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = nn.BCEWithLogitsLoss()
        for epoch in range(max_epochs):
            model.train()
            total_loss = 0.0
            for a_tok, b_tok, ma, mb, labels, _ in train_loader:
                a_tok, b_tok = a_tok.to(device), b_tok.to(device)
                ma, mb = ma.to(device), mb.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                logits = model(a_tok, b_tok, ma, mb)
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            val_logits, val_targets = predict(val_loader)
            val_probs = 1.0 / (1.0 + np.exp(-val_logits))
            df_preds = pd.DataFrame(val_probs, columns=[str(i) for i in range(num_labels)])
            df_targets = pd.DataFrame(val_targets, columns=[str(i) for i in range(num_labels)])
            metrics = compute_all_metrics(df_targets, df_preds, {}, {})
            val_macro_ap = metrics.get('macro_ap', 0.0)
            print(f"Epoch {epoch}: Loss={total_loss:.4f} Val AP={val_macro_ap:.4f}")
            if val_macro_ap > best_val_ap + 1e-4:
                best_val_ap = val_macro_ap
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(run_dir, "checkpoint_best.pt"))
            else:
                patience_counter += 1
            if patience_counter >= 5:
                print("Early stopping triggered.")
                break

    if isinstance(model, PrevalenceModel):
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))

    df_val_logits = pd.DataFrame(val_logits, columns=label_names)
    df_preds = pd.DataFrame(val_probs, columns=label_names)
    df_targets = pd.DataFrame(val_targets, columns=label_names)

    trainer.model_selected()
    
    # Reload best
    model.load_state_dict(torch.load(os.path.join(run_dir, "checkpoint_best.pt")))
    # Recompute validation logits from the selected checkpoint before calibration.
    val_logits, val_targets = predict(val_loader)
    df_val_logits = pd.DataFrame(val_logits, columns=label_names)
    df_targets = pd.DataFrame(val_targets, columns=label_names)
    
    # Calibration & Thresholds
    calibrated_val_probs, temps = calibrate_validation_logits(
        {"specific": df_val_logits}, {"specific": df_targets}
    )
    thresholds = select_thresholds(df_targets, calibrated_val_probs["specific"])

    def json_safe(value):
        if isinstance(value, dict):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, np.generic):
            return json_safe(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    with open(os.path.join(run_dir, "calibration.json"), "w") as f:
        json.dump(json_safe({"temperatures": temps, "input": "validation_raw_logits"}), f, indent=2)
    with open(os.path.join(run_dir, "thresholds.json"), "w") as f:
        json.dump(json_safe(thresholds), f, indent=2)

    # Persist validation outputs before freezing the validation decision.
    _persist_prediction_artifacts(
        run_dir,
        "validation",
        val_dataset,
        val_logits,
        val_targets,
        calibrated_val_probs["specific"].to_numpy(),
    )
    
    trainer.validation_frozen()
    
    # Test examples are not constructed until validation and model selection are frozen.
    test_dataset = ManifestPolypharmacyDataset.from_manifest(
        manifest_path, manifest, split="test", drug_features_path=experiment_config.get("drug_features_path", "artifacts/morgan_fingerprints.parquet")
    )
    if _label_names(test_dataset) != label_names:
        raise ValueError("test dataset label definition does not match train/validation")
    batch_size = experiment_config.get("batch_size", 32)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Test evaluation
    model.eval()
    test_logits, test_targets = predict(test_loader)
    
    test_probs = apply_temperature(test_logits, temps.get("specific", 1.0))
    df_test_preds = pd.DataFrame(test_probs, columns=label_names)
    df_test_targets = pd.DataFrame(test_targets, columns=label_names)

    _persist_prediction_artifacts(
        run_dir,
        "test",
        test_dataset,
        test_logits,
        test_targets,
        test_probs,
    )
    
    trainer.evaluate_test()
    
    final_metrics = compute_all_metrics(
        df_test_targets,
        df_test_preds,
        thresholds,
        train_prevalences,
    )
    _persist_per_label_metrics(run_dir, final_metrics)

    # StateGuardedTrainer verifies every artifact and writes completion.json
    # last.  The fallback keeps compatibility with lightweight test doubles
    # that implement the historic one-argument entry point.
    try:
        trainer.complete(final_metrics, required_artifacts=REQUIRED_RUN_ARTIFACTS)
    except TypeError as exc:
        if "required_artifacts" not in str(exc):
            raise
        trainer.complete(final_metrics)
    print(f"Run {run_id} complete.")

def main():
    args = parse_args()
    if args.all_benchmarks:
        manifest_paths = discover_benchmark_manifests(args.benchmarks_root, args.experiment)
        # Preflight above completes before this loop, so one invalid or
        # duplicate manifest cannot leave a partially executed cohort.
        for manifest_path in manifest_paths:
            run_single_experiment(args.experiment, str(manifest_path))
    else:
        run_single_experiment(args.experiment, args.benchmark)

if __name__ == "__main__":
    main()

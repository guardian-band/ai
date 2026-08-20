"""Validated training path for precomputed multimodal teacher/student inputs.

This entry point consumes upstream artifacts; it never trains MolFormer, a
molecular encoder, or PrimeKG/HGT.  ``--dry-run`` performs all validation and
model/dataset construction without creating a run directory or optimizer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
import torch.nn as nn

from run_experiment import (
    _label_names,
    _persist_per_label_metrics,
    _persist_prediction_artifacts,
)
from src.data.manifest_dataset import load_manifest_records
from src.data.precomputed_datasets import (
    MultimodalPairDataset,
    StudentPairDataset,
    validate_drug_coverage,
)
from src.evaluation.calibration import apply_temperature, calibrate_validation_logits
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.thresholds import select_thresholds
from src.features.cached_token_artifact import CachedTokenArtifact
from src.features.multimodal_feature_artifact import MultimodalFeatureArtifact
from src.models.factory import (
    UnsupportedModelConfiguration,
    create_model,
    load_experiment_config,
    validate_model_type,
)
from src.models.hierarchy import HierarchyLoss, HierarchyMapping, load_hierarchy_mapping
from src.models.multimodal_teacher_student import DistillationLoss
from src.training.engine import REQUIRED_RUN_ARTIFACTS, StateGuardedTrainer, TrainerState, verify_manifest
from src.training.reproducibility import collect_environment_info, set_global_seed


def choose_device() -> torch.device:
    """Select CUDA, then Apple MPS, then CPU for actual training."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimizer(model: nn.Module, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    learning_rate = config.get("learning_rate")
    weight_decay = config.get("weight_decay")
    _validate_finite_number(learning_rate, "learning_rate", positive=True)
    _validate_finite_number(weight_decay, "weight_decay", nonnegative=True)
    return torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )


def _validate_finite_number(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    if positive and float(value) <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and float(value) < 0:
        raise ValueError(f"{name} must be non-negative")


def validate_training_config(config: Mapping[str, Any]) -> None:
    """Validate every training control before any run directory is created."""

    model_type = config.get("model_type")
    required_ints = ("batch_size", "epochs", "patience")
    for key in required_ints:
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    _validate_finite_number(config.get("learning_rate"), "learning_rate", positive=True)
    for key in ("weight_decay", "hierarchy_weight"):
        _validate_finite_number(config.get(key), key, nonnegative=True)
    for key in ("supervised_weight", "distillation_weight"):
        if key in config or model_type == "distilled_pair_student":
            _validate_finite_number(config.get(key), key, nonnegative=True)
    if "distillation_temperature" in config or model_type == "distilled_pair_student":
        _validate_finite_number(
            config.get("distillation_temperature"), "distillation_temperature", positive=True
        )


def _require_defined_macro_ap(value: float) -> float:
    if not math.isfinite(float(value)):
        raise ValueError("validation macro AP is undefined; validation labels must contain both classes")
    return float(value)


def _validate_validation_support(dataset: Any) -> None:
    records = getattr(dataset, "records", None)
    if not records:
        raise ValueError("validation macro AP is undefined: validation split is empty")
    targets = np.asarray([record["labels"] for record in records], dtype=float)
    if targets.ndim != 2 or not any(np.unique(targets[:, index]).size == 2 for index in range(targets.shape[1])):
        raise ValueError("validation macro AP is undefined: no validation label contains both classes")


def _manifest_label_contract(
    manifest_path: Path, manifest: Mapping[str, Any]
) -> tuple[tuple[str, ...], int]:
    parsed_by_split = {
        split: load_manifest_records(manifest_path, manifest, split)
        for split in ("train", "validation", "test")
    }
    orders = {
        split: tuple(str(label["cui"]) for label in parsed.labels)
        for split, parsed in parsed_by_split.items()
    }
    expected = orders["train"]
    for split, order in orders.items():
        if order != expected:
            raise ValueError(f"manifest label order mismatch between train and {split}")
    if not expected:
        raise ValueError("manifest must define at least one specific label")
    return expected, len(expected)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required_path(config_path: Path, config: Mapping[str, Any], key: str) -> Path:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"config requires non-empty {key}")
    path = Path(value)
    return path if path.is_absolute() else config_path.parent / path


def _required_hash(config: Mapping[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"config requires canonical lowercase SHA-256 {key}")
    return value


def _verify_source(path: Path, expected: str, name: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {name}: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{name} hash mismatch: {actual} != {expected}")
    return actual


def _compatibility_check(artifact: MultimodalFeatureArtifact, manifest: Mapping[str, Any]) -> None:
    compatibility = artifact.metadata["manifest_compatibility"]
    required = ("benchmark_id", "scenario", "seed", "manifest_hash")
    missing = [key for key in required if key not in compatibility]
    if missing:
        raise ValueError(f"feature artifact manifest compatibility missing: {', '.join(missing)}")
    for key in required:
        if compatibility[key] != manifest.get(key):
            raise ValueError(f"feature artifact manifest compatibility mismatch for {key}")


def _feature_schema_check(
    artifact: MultimodalFeatureArtifact,
    config: Mapping[str, Any],
    *,
    source_name: str,
) -> None:
    expected = {
        "morgan_dim": artifact.morgan.shape[1],
        "molformer_dim": artifact.molformer_tokens.shape[2],
        "mpnn_dim": artifact.mpnn_tokens.shape[2],
        "kg_dim": artifact.kg_tokens.shape[2],
        "molformer_token_count": artifact.molformer_tokens.shape[1],
        "mpnn_token_count": artifact.mpnn_tokens.shape[1],
        "kg_token_count": artifact.kg_tokens.shape[1],
    }
    mismatches = {
        key: (config[key], actual)
        for key, actual in expected.items()
        if key in config and config[key] != actual
    }
    if mismatches:
        details = ", ".join(
            f"{key}={configured} (artifact={actual})"
            for key, (configured, actual) in sorted(mismatches.items())
        )
        raise ValueError(f"{source_name} schema mismatch: {details}")


def _freeze_teacher(model: nn.Module) -> nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@dataclass
class PreflightPlan:
    config_path: Path
    config: dict[str, Any]
    manifest_path: Path
    manifest: dict[str, Any]
    hierarchy: HierarchyMapping
    model: nn.Module
    train_dataset: Any
    validation_dataset: Any
    feature_artifact: MultimodalFeatureArtifact | None = None
    cached_artifact: CachedTokenArtifact | None = None
    teacher_model: nn.Module | None = None
    teacher_train_dataset: Any | None = None
    teacher_validation_dataset: Any | None = None
    source_hashes: dict[str, str] | None = None


def _load_hierarchy(config_path: Path, config: Mapping[str, Any]) -> tuple[HierarchyMapping, str, Path]:
    path = _required_path(config_path, config, "hierarchy_path")
    expected = _required_hash(config, "hierarchy_sha256")
    _verify_source(path, expected, "hierarchy artifact")
    return load_hierarchy_mapping(path), expected, path


def _feature_dataset_pair(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    config_path: Path,
    config: Mapping[str, Any],
    hierarchy_path: Path,
) -> tuple[MultimodalFeatureArtifact, MultimodalPairDataset, MultimodalPairDataset, dict[str, str]]:
    feature_path = _required_path(config_path, config, "feature_artifact_path")
    feature_hash = _required_hash(config, "feature_artifact_sha256")
    _verify_source(feature_path, feature_hash, "multimodal feature artifact")
    artifact = MultimodalFeatureArtifact.load(feature_path)
    _compatibility_check(artifact, manifest)
    _feature_schema_check(artifact, config, source_name="multimodal feature artifact")
    train = MultimodalPairDataset.from_manifest(manifest_path, manifest, "train", artifact, hierarchy_path)
    validation = MultimodalPairDataset.from_manifest(manifest_path, manifest, "validation", artifact, hierarchy_path)
    test_records = load_manifest_records(manifest_path, manifest, "test")
    validate_drug_coverage(test_records, artifact.drug_ids, split="test")
    load_hierarchy_mapping(
        hierarchy_path,
        selected_specific_cuis=[str(label["cui"]) for label in test_records.labels],
    )
    return artifact, train, validation, {"feature_artifact": feature_hash}


def preflight_experiment(
    experiment_config_path: str | Path,
    manifest_path: str | Path,
) -> PreflightPlan:
    """Validate all source contracts and construct only train/validation data."""

    config_path = Path(experiment_config_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    manifest = verify_manifest(str(manifest_path))
    config = load_experiment_config(config_path)
    model_type = validate_model_type(config)
    if model_type not in {"multimodal_teacher", "distilled_pair_student"}:
        raise UnsupportedModelConfiguration(
            "run_precomputed_experiment supports only multimodal_teacher and distilled_pair_student"
        )
    validate_training_config(config)
    hierarchy, hierarchy_hash, hierarchy_path = _load_hierarchy(config_path, config)
    label_order, num_labels = _manifest_label_contract(manifest_path, manifest)
    source_hashes = {
        "experiment_config": _sha256(config_path),
        "manifest_file": _sha256(manifest_path),
        "manifest_claimed_hash": str(manifest["manifest_hash"]),
        "hierarchy": hierarchy_hash,
    }

    if model_type == "multimodal_teacher":
        feature, train, validation, hashes = _feature_dataset_pair(
            manifest_path, manifest, config_path, config, hierarchy_path
        )
        model = create_model(config, num_labels=num_labels)
        if model.num_specific != num_labels:
            raise ValueError("model num_specific does not match manifest label order")
        if model.num_organ != len(hierarchy.organ_order):
            raise ValueError("model num_organ does not match hierarchy organ_order")
        _validate_validation_support(validation)
        source_hashes.update(hashes)
        return PreflightPlan(
            config_path, config, manifest_path, manifest, hierarchy, model, train, validation,
            feature_artifact=feature, source_hashes=source_hashes,
        )

    cached_path = _required_path(config_path, config, "cached_token_artifact_path")
    cached_hash = _required_hash(config, "cached_token_artifact_sha256")
    _verify_source(cached_path, cached_hash, "cached-token artifact")
    cached = CachedTokenArtifact.load(cached_path)
    student_train = StudentPairDataset.from_manifest(manifest_path, manifest, "train", cached, hierarchy_path)
    student_validation = StudentPairDataset.from_manifest(manifest_path, manifest, "validation", cached, hierarchy_path)
    student_test_records = load_manifest_records(manifest_path, manifest, "test")
    validate_drug_coverage(student_test_records, cached.drug_ids, split="test")
    load_hierarchy_mapping(
        hierarchy_path,
        selected_specific_cuis=[str(label["cui"]) for label in student_test_records.labels],
    )
    teacher_config_path = _required_path(config_path, config, "teacher_config_path")
    teacher_config_hash = _required_hash(config, "teacher_config_sha256")
    _verify_source(teacher_config_path, teacher_config_hash, "teacher config")
    teacher_config = load_experiment_config(teacher_config_path)
    if validate_model_type(teacher_config) != "multimodal_teacher":
        raise ValueError("teacher_config_path must declare multimodal_teacher")
    teacher_feature, teacher_train, teacher_validation, feature_hashes = _feature_dataset_pair(
        manifest_path, manifest, teacher_config_path, teacher_config, hierarchy_path
    )
    teacher_checkpoint_path = _required_path(config_path, config, "teacher_checkpoint_path")
    teacher_checkpoint_hash = _required_hash(config, "teacher_checkpoint_sha256")
    _verify_source(teacher_checkpoint_path, teacher_checkpoint_hash, "teacher checkpoint")
    if cached.metadata.get("teacher_checkpoint_hash") != teacher_checkpoint_hash:
        raise ValueError("cached-token artifact teacher checkpoint hash does not match config")
    if cached.metadata.get("teacher_config_hash") != teacher_config_hash:
        raise ValueError("cached-token artifact teacher config hash does not match config")
    if cached.metadata.get("modality_provenance_hash") != feature_hashes["feature_artifact"]:
        raise ValueError(
            "cached-token artifact modality provenance does not match the teacher feature artifact"
        )
    teacher = create_model(teacher_config, num_labels=num_labels)
    checkpoint = torch.load(teacher_checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("teacher checkpoint must contain a state-dict mapping")
    teacher.load_state_dict(checkpoint)
    _freeze_teacher(teacher)
    model = create_model(config, num_labels=num_labels)
    if model.num_specific != num_labels or teacher.num_specific != num_labels:
        raise ValueError("teacher/student num_specific does not match manifest label order")
    if model.num_organ != len(hierarchy.organ_order) or teacher.num_organ != len(hierarchy.organ_order):
        raise ValueError("teacher/student num_organ does not match hierarchy organ_order")
    if (
        model.token_count != cached.token_count
        or model.token_dim != cached.token_dim
        or teacher.cache_token_count != cached.token_count
        or teacher.cache_token_dim != cached.token_dim
    ):
        raise ValueError("student config and cached-token artifact schemas do not match")
    _validate_validation_support(student_validation)
    if [record["pair_id"] for record in teacher_train.records] != [record["pair_id"] for record in student_train.records]:
        raise ValueError("teacher and student training pair order does not match")
    if [record["pair_id"] for record in teacher_validation.records] != [record["pair_id"] for record in student_validation.records]:
        raise ValueError("teacher and student validation pair order does not match")
    source_hashes.update(feature_hashes)
    source_hashes.update({"cached_tokens": cached_hash, "teacher_config": teacher_config_hash, "teacher_checkpoint": teacher_checkpoint_hash})
    return PreflightPlan(
        config_path, config, manifest_path, manifest, hierarchy, model, student_train, student_validation,
        cached_artifact=cached, teacher_model=teacher, teacher_train_dataset=teacher_train,
        teacher_validation_dataset=teacher_validation, source_hashes=source_hashes,
    )


def _macro_ap(logits: np.ndarray, targets: np.ndarray) -> float:
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    values = []
    for index in range(targets.shape[1]):
        if len(np.unique(targets[:, index])) == 2:
            values.append(average_precision_score(targets[:, index], probabilities[:, index]))
    return float(np.mean(values)) if values else float("nan")


def _teacher_or_student_batch(batch: tuple[Any, ...], device: torch.device, student: bool):
    if student:
        tokens_a, tokens_b, availability_a, availability_b, organ, specific, *_ = batch
        return (
            tokens_a.to(device), tokens_b.to(device), availability_a.to(device), availability_b.to(device),
            organ.to(device), specific.to(device),
        )
    drug_a, drug_b, organ, specific, *_ = batch
    drug_a = {key: value.to(device) for key, value in drug_a.items()}
    drug_b = {key: value.to(device) for key, value in drug_b.items()}
    return drug_a, drug_b, organ.to(device), specific.to(device)


def _predict(model: nn.Module, loader: DataLoader, device: torch.device, student: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, targets = [], []
    with torch.inference_mode():
        for batch in loader:
            values = _teacher_or_student_batch(batch, device, student)
            if student:
                output = model(*values[:2], values[2], values[3])
            else:
                output = model(values[0], values[1])
            logits.append(output.specific_logits.cpu().numpy())
            targets.append(values[-1].cpu().numpy())
    return np.concatenate(logits), np.concatenate(targets)


def _construct_test_dataset_after_freeze(
    plan: PreflightPlan, student_mode: bool, trainer: StateGuardedTrainer
) -> StudentPairDataset | MultimodalPairDataset:
    if trainer.state != TrainerState.VALIDATION_FROZEN:
        raise RuntimeError("test dataset construction requires validation_frozen state")
    hierarchy_path = _required_path(plan.config_path, plan.config, "hierarchy_path")
    if student_mode:
        return StudentPairDataset.from_manifest(
            plan.manifest_path, plan.manifest, "test", plan.cached_artifact, hierarchy_path
        )
    return MultimodalPairDataset.from_manifest(
        plan.manifest_path, plan.manifest, "test", plan.feature_artifact, hierarchy_path
    )


def run_precomputed_experiment(
    experiment_config_path: str | Path,
    manifest_path: str | Path,
    *,
    dry_run: bool = False,
) -> PreflightPlan | None:
    plan = preflight_experiment(experiment_config_path, manifest_path)
    if dry_run:
        return plan
    set_global_seed(int(plan.manifest["seed"]))
    device = choose_device()
    model = plan.model.to(device)
    student_mode = plan.config["model_type"] == "distilled_pair_student"
    run_id = f"{plan.config['model_type']}__{plan.manifest.get('scenario')}__seed_{plan.manifest['seed']}__{plan.manifest['manifest_hash']}"
    run_dir = Path("artifacts/runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    trainer = StateGuardedTrainer(str(run_dir))
    (run_dir / "environment.json").write_text(json.dumps(collect_environment_info(), indent=2))
    labels = _label_names(plan.train_dataset)
    train_prevalences = {
        label: float(np.mean([record["labels"][index] for record in plan.train_dataset.records]))
        for index, label in enumerate(labels)
    }
    resolved = dict(plan.config)
    resolved.update(
        {
            "run_id": run_id,
            "benchmark_id": plan.manifest["benchmark_id"],
            "scenario": plan.manifest["scenario"],
            "seed": int(plan.manifest["seed"]),
            "manifest_hash": plan.manifest["manifest_hash"],
            "train_prevalences": train_prevalences,
            "source_hashes": plan.source_hashes,
            "device": str(device),
        }
    )
    (run_dir / "config.resolved.json").write_text(json.dumps(resolved, indent=2))
    train_loader = DataLoader(plan.train_dataset, batch_size=int(plan.config["batch_size"]), shuffle=not student_mode)
    validation_loader = DataLoader(plan.validation_dataset, batch_size=int(plan.config["batch_size"]))
    teacher_train_loader = (
        DataLoader(plan.teacher_train_dataset, batch_size=int(plan.config["batch_size"]))
        if student_mode
        else None
    )
    optimizer = build_optimizer(model, plan.config)
    hierarchy_loss = HierarchyLoss(
        {index: plan.hierarchy.specific_to_organ[str(label["cui"])] for index, label in enumerate(plan.train_dataset.labels)},
        num_specific=len(plan.train_dataset.labels), num_organ=len(plan.hierarchy.organ_order),
    )
    teacher_model = plan.teacher_model.to(device) if plan.teacher_model is not None else None
    if teacher_model is not None:
        _freeze_teacher(teacher_model).to(device)
    trainer.begin_training()
    patience = int(plan.config["patience"])
    if patience <= 0:
        raise ValueError("patience must be a positive integer")
    best_ap = -float("inf")
    best_epoch = 0
    stale = 0
    epochs = int(plan.config["epochs"])
    model_name = "student" if student_mode else "teacher"
    print(
        f"[training] model={model_name} epochs={epochs} patience={patience} "
        f"batches_per_epoch={len(train_loader)}",
        flush=True,
    )
    for _epoch in range(epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_example_count = 0
        batches = zip(train_loader, teacher_train_loader) if student_mode else ((batch, None) for batch in train_loader)
        for batch, teacher_batch in batches:
            values = _teacher_or_student_batch(batch, device, student_mode)
            optimizer.zero_grad()
            if student_mode:
                output = model(*values[:2], values[2], values[3])
                with torch.no_grad():
                    teacher_values = _teacher_or_student_batch(teacher_batch, device, False)
                    teacher_output = teacher_model(teacher_values[0], teacher_values[1])
                loss = DistillationLoss(
                    temperature=float(plan.config["distillation_temperature"]),
                    supervised_weight=float(plan.config["supervised_weight"]),
                    distillation_weight=float(plan.config["distillation_weight"]),
                    hierarchy_weight=float(plan.config["hierarchy_weight"]),
                    hierarchy_loss=hierarchy_loss,
                )(
                    output,
                    teacher_output,
                    organ_targets=values[-2],
                    specific_targets=values[-1],
                )
            else:
                output = model(values[0], values[1])
                loss = (
                    nn.functional.binary_cross_entropy_with_logits(output.organ_logits, values[-2])
                    + nn.functional.binary_cross_entropy_with_logits(output.specific_logits, values[-1])
                    + float(plan.config["hierarchy_weight"])
                    * hierarchy_loss(output.specific_logits, output.organ_logits)
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at epoch {_epoch + 1}")
            loss.backward()
            optimizer.step()
            batch_example_count = int(values[-1].shape[0])
            epoch_loss_sum += float(loss.detach().cpu()) * batch_example_count
            epoch_example_count += batch_example_count
        if epoch_example_count == 0:
            raise RuntimeError(f"no training examples were processed at epoch {_epoch + 1}")
        train_loss = epoch_loss_sum / epoch_example_count
        val_logits, val_targets = _predict(model, validation_loader, device, student_mode)
        val_ap = _require_defined_macro_ap(_macro_ap(val_logits, val_targets))
        checkpoint_saved = False
        if not (run_dir / "checkpoint_best.pt").is_file() or (
            np.isfinite(val_ap) and val_ap > best_ap
        ):
            best_ap = val_ap
            best_epoch = _epoch + 1
            stale = 0
            torch.save(model.state_dict(), run_dir / "checkpoint_best.pt")
            checkpoint_saved = True
        else:
            stale += 1
        print(
            f"[epoch {_epoch + 1:02d}/{epochs:02d}] model={model_name} "
            f"train_loss={train_loss:.6f} val_macro_auprc={val_ap:.6f} "
            f"best_val_macro_auprc={best_ap:.6f} patience={stale}/{patience} "
            f"checkpoint={'saved' if checkpoint_saved else 'kept'}",
            flush=True,
        )
        if stale >= patience:
            print(
                f"[early-stop] model={model_name} epoch={_epoch + 1} "
                f"best_epoch={best_epoch}",
                flush=True,
            )
            break
    if not (run_dir / "checkpoint_best.pt").is_file():
        raise RuntimeError("no validation-selected checkpoint was produced")
    model.load_state_dict(torch.load(run_dir / "checkpoint_best.pt", map_location=device, weights_only=True))
    print(
        f"[selected] model={model_name} best_epoch={best_epoch} "
        f"best_val_macro_auprc={best_ap:.6f}",
        flush=True,
    )
    trainer.model_selected()
    print("[phase] validation calibration and threshold selection", flush=True)
    val_logits, val_targets = _predict(model, validation_loader, device, student_mode)
    labels = _label_names(plan.validation_dataset)
    val_logits_frame = pd.DataFrame(val_logits, columns=labels)
    val_targets_frame = pd.DataFrame(val_targets, columns=labels)
    calibrated, temperatures = calibrate_validation_logits({"specific": val_logits_frame}, {"specific": val_targets_frame})
    thresholds = select_thresholds(val_targets_frame, calibrated["specific"])
    _persist_prediction_artifacts(str(run_dir), "validation", plan.validation_dataset, val_logits, val_targets, calibrated["specific"].to_numpy())
    (run_dir / "calibration.json").write_text(json.dumps({"temperatures": temperatures, "input": "validation_raw_logits"}, indent=2))
    (run_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    trainer.validation_frozen()
    print("[phase] test inference and metric computation", flush=True)
    test_dataset = _construct_test_dataset_after_freeze(plan, student_mode, trainer)
    test_loader = DataLoader(test_dataset, batch_size=int(plan.config["batch_size"]))
    test_logits, test_targets = _predict(model, test_loader, device, student_mode)
    test_probs = apply_temperature(test_logits, temperatures["specific"])
    _persist_prediction_artifacts(str(run_dir), "test", test_dataset, test_logits, test_targets, test_probs)
    trainer.evaluate_test()
    metrics = compute_all_metrics(pd.DataFrame(test_targets, columns=labels), pd.DataFrame(test_probs, columns=labels), thresholds, train_prevalences)
    print(
        f"[test] model={model_name} micro_auprc={metrics['micro_ap']:.6f} "
        f"macro_auprc={metrics['macro_ap']:.6f} "
        f"micro_auroc={metrics['micro_auroc']:.6f} "
        f"macro_auroc={metrics['macro_auroc']:.6f} "
        f"brier={metrics['brier_score']:.6f} ece={metrics['ece_15']:.6f}",
        flush=True,
    )
    _persist_per_label_metrics(str(run_dir), metrics)
    trainer.complete(metrics, required_artifacts=REQUIRED_RUN_ARTIFACTS)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_precomputed_experiment(args.experiment, args.benchmark, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

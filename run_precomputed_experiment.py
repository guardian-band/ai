"""Validated training path for precomputed multimodal teacher/student inputs.

This entry point consumes upstream artifacts; it never trains MolFormer, a
molecular encoder, or PrimeKG/HGT.  ``--dry-run`` performs all validation and
model/dataset construction without creating a run directory or optimizer.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
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
    derive_run_model_id,
    load_experiment_config,
    validate_enabled_modalities,
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


def build_optimizer(
    model: nn.Module,
    config: Mapping[str, Any],
    parameters: Any | None = None,
) -> torch.optim.Optimizer:
    learning_rate = config.get("learning_rate")
    weight_decay = config.get("weight_decay")
    _validate_finite_number(learning_rate, "learning_rate", positive=True)
    _validate_finite_number(weight_decay, "weight_decay", nonnegative=True)
    return torch.optim.AdamW(
        model.parameters() if parameters is None else parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )


def _validate_finite_number(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    if positive and float(value) <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and float(value) < 0:
        raise ValueError(f"{name} must be non-negative")


def _derive_run_model_id(config: Mapping[str, Any]) -> str:
    """Expose the provenance-safe run model ID used by the runner."""

    return derive_run_model_id(config)


def _candidate_grid(config: Mapping[str, Any]) -> list[tuple[float, float]]:
    """Return the deduplicated distillation candidate grid.

    The supervised-only candidate is mandatory even when omitted from YAML.
    Candidate order is kept deterministic; validation AP determines selection.
    """

    raw_weights = config.get(
        "distillation_weight_candidates",
        [config.get("distillation_weight", 1.0)],
    )
    raw_temperatures = config.get(
        "distillation_temperature_candidates",
        [config.get("distillation_temperature", 2.0)],
    )
    if not isinstance(raw_weights, (list, tuple)) or not raw_weights:
        raise ValueError("distillation_weight_candidates must be a non-empty list")
    if not isinstance(raw_temperatures, (list, tuple)) or not raw_temperatures:
        raise ValueError("distillation_temperature_candidates must be a non-empty list")
    weights = [0.0]
    for value in raw_weights:
        _validate_finite_number(value, "distillation weight candidate", nonnegative=True)
        if float(value) not in weights:
            weights.append(float(value))
    temperatures: list[float] = []
    for value in raw_temperatures:
        _validate_finite_number(value, "distillation temperature candidate", positive=True)
        if float(value) not in temperatures:
            temperatures.append(float(value))
    # Temperature has no effect when the distillation weight is zero, so the
    # supervised-only candidate is evaluated exactly once.
    candidates = [(0.0, min(temperatures))]
    candidates.extend(
        (weight, temperature)
        for weight in weights
        if weight != 0.0
        for temperature in temperatures
    )
    return candidates


def _select_distillation_candidate(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("distillation candidate evaluation produced no results")
    normalized = []
    for result in results:
        ap = result.get("validation_macro_ap")
        _validate_finite_number(ap, "distillation candidate validation macro AP")
        weight = result.get("distillation_weight")
        temperature = result.get("temperature", result.get("distillation_temperature"))
        _validate_finite_number(weight, "distillation candidate weight", nonnegative=True)
        _validate_finite_number(temperature, "distillation candidate temperature", positive=True)
        normalized_result = dict(result)
        normalized_result["temperature"] = float(temperature)
        normalized.append(normalized_result)
    return min(
        normalized,
        key=lambda item: (
            -float(item["validation_macro_ap"]),
            float(item["distillation_weight"]),
            float(item["temperature"]),
        ),
    )


def _assert_batch_alignment(student_pair_ids: Any, teacher_pair_ids: Any) -> None:
    student_ids = tuple(str(value) for value in student_pair_ids)
    teacher_ids = tuple(str(value) for value in teacher_pair_ids)
    if len(student_ids) != len(teacher_ids):
        raise ValueError(
            "student/teacher pair_id batch length mismatch; refusing unequal loader exhaustion"
        )
    if student_ids != teacher_ids:
        raise ValueError("student/teacher pair_id batch order mismatch")


def _batch_pair_ids(batch: tuple[Any, ...], *, student: bool) -> tuple[str, ...]:
    index = 6 if student else 4
    if len(batch) <= index:
        raise ValueError("training batch does not contain pair_id values")
    values = batch[index]
    if isinstance(values, str):
        return (values,)
    return tuple(str(value) for value in values)


def _aligned_student_teacher_batches(
    student_loader: DataLoader, teacher_loader: DataLoader
):
    student_iterator = iter(student_loader)
    teacher_iterator = iter(teacher_loader)
    while True:
        try:
            student_batch = next(student_iterator)
        except StopIteration:
            try:
                next(teacher_iterator)
            except StopIteration:
                return
            raise ValueError("student/teacher loader length mismatch")
        try:
            teacher_batch = next(teacher_iterator)
        except StopIteration as exc:
            raise ValueError("student/teacher loader length mismatch") from exc
        _assert_batch_alignment(
            _batch_pair_ids(student_batch, student=True),
            _batch_pair_ids(teacher_batch, student=False),
        )
        yield student_batch, teacher_batch


def _validate_student_teacher_ap(
    student_ap: float, teacher_ap: float, max_drop: float = 0.005
) -> None:
    _validate_finite_number(student_ap, "student validation macro AP")
    _validate_finite_number(teacher_ap, "teacher validation macro AP")
    _validate_finite_number(max_drop, "student_max_validation_ap_drop", nonnegative=True)
    if student_ap < teacher_ap - float(max_drop):
        raise ValueError(
            "Student validation macro AP is below selected Teacher by more than "
            f"student_max_validation_ap_drop ({student_ap} < {teacher_ap} - {max_drop})"
        )


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
    if model_type == "multimodal_teacher":
        validate_enabled_modalities(config.get("enabled_modalities"))
        if "modality_dropout" in config:
            _validate_finite_number(config["modality_dropout"], "modality_dropout", nonnegative=True)
            if float(config["modality_dropout"]) >= 1.0:
                raise ValueError("modality_dropout must be less than 1")
        if "modality_summary_token_count" in config:
            value = config["modality_summary_token_count"]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("modality_summary_token_count must be a positive integer")
        _validate_finite_number(
            config.get("promotion_min_delta", 0.002),
            "promotion_min_delta",
            nonnegative=True,
        )
    if model_type == "distilled_pair_student":
        _candidate_grid(config)
        _validate_finite_number(
            config.get("student_max_validation_ap_drop", 0.005),
            "student_max_validation_ap_drop",
            nonnegative=True,
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


def _manifest_label_provenance(
    manifest_path: Path, manifest: Mapping[str, Any], label_order: tuple[str, ...]
) -> dict[str, str]:
    raw_path = manifest.get("labels_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("Manifest must define a non-empty labels_path")
    labels_path = Path(raw_path)
    if not labels_path.is_absolute():
        labels_path = manifest_path.parent / labels_path
    labels_artifact_hash = _sha256(labels_path)
    labels_order_hash = hashlib.sha256(
        json.dumps(list(label_order), separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    return {
        "labels_artifact": labels_artifact_hash,
        "labels_order": labels_order_hash,
    }


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


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON with a replace-in-place commit so partial metadata is impossible."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _required_run_artifacts(*, student_mode: bool) -> tuple[str, ...]:
    selection_name = (
        "student_distillation_selection.json"
        if student_mode
        else "teacher_validation_selection.json"
    )
    return REQUIRED_RUN_ARTIFACTS + (selection_name,)


def _verify_source(path: Path, expected: str, name: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {name}: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{name} hash mismatch: {actual} != {expected}")
    return actual


def _validate_teacher_selection(
    path: Path, *, expected_hash: str | None = None
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing teacher validation selection: {path}")
    actual_hash = _sha256(path)
    if expected_hash is not None and actual_hash != expected_hash:
        raise ValueError(
            f"teacher validation selection hash mismatch: {actual_hash} != {expected_hash}"
        )
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"teacher validation selection is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("teacher validation selection must be a JSON object")
    selected = payload.get("selected")
    if (
        not isinstance(selected, dict)
        or not isinstance(selected.get("mode"), str)
        or selected.get("mode") not in {"baseline", "fused"}
    ):
        raise ValueError("teacher validation selection mode must be exactly baseline or fused")
    _validate_finite_number(selected.get("macro_ap"), "teacher selected validation macro AP")
    return payload


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
    # Model construction below initializes randomized parameters.  Seed before
    # creating the model so runs carrying the same manifest seed start from the
    # same weights instead of inheriting process-specific RNG state.
    set_global_seed(int(manifest["seed"]))
    config = load_experiment_config(config_path)
    model_type = validate_model_type(config)
    derived_run_model_id = _derive_run_model_id(config)
    if "run_model_id" in config and config["run_model_id"] != derived_run_model_id:
        raise ValueError(
            f"run_model_id is derived as {derived_run_model_id}; supplied value is not allowed"
        )
    if model_type not in {"multimodal_teacher", "distilled_pair_student"}:
        raise UnsupportedModelConfiguration(
            "run_precomputed_experiment supports only multimodal_teacher and distilled_pair_student"
        )
    validate_training_config(config)
    hierarchy, hierarchy_hash, hierarchy_path = _load_hierarchy(config_path, config)
    label_order, num_labels = _manifest_label_contract(manifest_path, manifest)
    label_provenance = _manifest_label_provenance(manifest_path, manifest, label_order)
    source_hashes = {
        "experiment_config": _sha256(config_path),
        "manifest_file": _sha256(manifest_path),
        "manifest_claimed_hash": str(manifest["manifest_hash"]),
        "hierarchy": hierarchy_hash,
    }
    source_hashes.update(label_provenance)
    for config_key, source_key in (
        ("labels_artifact_sha256", "labels_artifact"),
        ("labels_order_sha256", "labels_order"),
    ):
        configured_hash = config.get(config_key)
        if configured_hash is not None and configured_hash != source_hashes[source_key]:
            raise ValueError(f"{config_key} does not match manifest provenance")
    configured_manifest_hash = config.get("manifest_hash")
    if configured_manifest_hash is not None and configured_manifest_hash != manifest["manifest_hash"]:
        raise ValueError("config manifest_hash does not match manifest")

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
    teacher_selection_path = _required_path(config_path, config, "teacher_selection_path")
    teacher_selection_hash = _required_hash(config, "teacher_selection_sha256")
    teacher_selection = _validate_teacher_selection(
        teacher_selection_path, expected_hash=teacher_selection_hash
    )
    selected_mode = config.get("teacher_selected_mode")
    if not isinstance(selected_mode, str) or selected_mode not in {"baseline", "fused"}:
        raise ValueError("config teacher_selected_mode must be exactly baseline or fused")
    if teacher_selection["selected"]["mode"] != selected_mode:
        raise ValueError("teacher selection mode does not match Student config")
    if cached.metadata.get("teacher_selection_hash") != teacher_selection_hash:
        raise ValueError("cached-token artifact teacher selection hash does not match config")
    if cached.metadata.get("teacher_selected_mode") != selected_mode:
        raise ValueError("cached-token artifact teacher selected mode does not match config")
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
    teacher.set_training_stage(selected_mode)
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
    source_hashes.update(
        {
            "cached_tokens": cached_hash,
            "teacher_config": teacher_config_hash,
            "teacher_checkpoint": teacher_checkpoint_hash,
            "teacher_selection": teacher_selection_hash,
        }
    )
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


def _select_teacher_promotion(
    baseline_ap: float, fused_ap: float, promotion_min_delta: float = 0.002
) -> str:
    """Promote auxiliary fusion only for a strict validation AP improvement."""

    _validate_finite_number(baseline_ap, "baseline validation macro AP")
    _validate_finite_number(fused_ap, "fused validation macro AP")
    _validate_finite_number(promotion_min_delta, "promotion_min_delta", nonnegative=True)
    return "fused" if fused_ap > baseline_ap + float(promotion_min_delta) else "baseline"


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


def _predict_teacher_variants(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, dict[str, list[Any]]]]:
    if not hasattr(model, "validation_variants"):
        raise TypeError("teacher model must expose validation_variants")
    model.eval()
    logits_by_variant: dict[str, list[np.ndarray]] = {}
    targets: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            drug_a, drug_b, _, specific = _teacher_or_student_batch(batch, device, False)
            variants = model.validation_variants(drug_a, drug_b)
            for name, output in variants.items():
                logits_by_variant.setdefault(name, []).append(output.specific_logits.cpu().numpy())
            targets.append(specific.cpu().numpy())
    target_array = np.concatenate(targets)
    predictions = {
        name: (np.concatenate(values), target_array) for name, values in logits_by_variant.items()
    }
    diagnostics = model.diagnostics() if hasattr(model, "diagnostics") else {}
    serializable = {
        category: {name: value.detach().cpu().tolist() for name, value in values.items()}
        for category, values in diagnostics.items()
    }
    return predictions, serializable


def _teacher_supervised_loss(
    output: Any,
    organ_targets: torch.Tensor,
    specific_targets: torch.Tensor,
    hierarchy_loss: HierarchyLoss,
    hierarchy_weight: float,
) -> torch.Tensor:
    return (
        nn.functional.binary_cross_entropy_with_logits(output.organ_logits, organ_targets)
        + nn.functional.binary_cross_entropy_with_logits(output.specific_logits, specific_targets)
        + hierarchy_weight * hierarchy_loss(output.specific_logits, output.organ_logits)
    )


def _train_student_candidate(
    model: nn.Module,
    initial_state: Mapping[str, torch.Tensor],
    student_loader: DataLoader,
    teacher_loader: DataLoader,
    validation_loader: DataLoader,
    teacher_model: nn.Module,
    device: torch.device,
    config: Mapping[str, Any],
    hierarchy_loss: HierarchyLoss,
    distillation_weight: float,
    temperature: float,
) -> tuple[float, dict[str, torch.Tensor]]:
    model.load_state_dict(initial_state)
    optimizer = build_optimizer(model, config)
    best_ap = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    epochs = int(config["epochs"])
    print(
        f"[training] model=student distillation_weight={distillation_weight:g} "
        f"temperature={temperature:g} epochs={epochs} patience={config['patience']} "
        f"batches_per_epoch={len(student_loader)}",
        flush=True,
    )
    for _epoch in range(epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_example_count = 0
        for batch, teacher_batch in _aligned_student_teacher_batches(
            student_loader, teacher_loader
        ):
            values = _teacher_or_student_batch(batch, device, True)
            teacher_values = _teacher_or_student_batch(teacher_batch, device, False)
            optimizer.zero_grad()
            output = model(*values[:2], values[2], values[3])
            with torch.no_grad():
                teacher_output = teacher_model(teacher_values[0], teacher_values[1])
            loss = DistillationLoss(
                temperature=temperature,
                supervised_weight=float(config["supervised_weight"]),
                distillation_weight=distillation_weight,
                hierarchy_weight=float(config["hierarchy_weight"]),
                hierarchy_loss=hierarchy_loss,
            )(
                output,
                teacher_output,
                organ_targets=values[-2],
                specific_targets=values[-1],
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite student training loss")
            loss.backward()
            optimizer.step()
            batch_example_count = int(values[-1].shape[0])
            epoch_loss_sum += float(loss.detach().cpu()) * batch_example_count
            epoch_example_count += batch_example_count
        if epoch_example_count == 0:
            raise RuntimeError("no student training examples were processed")
        val_logits, val_targets = _predict(model, validation_loader, device, True)
        val_ap = _require_defined_macro_ap(_macro_ap(val_logits, val_targets))
        checkpoint_saved = False
        if val_ap > best_ap:
            best_ap = val_ap
            best_epoch = _epoch + 1
            stale = 0
            best_state = copy.deepcopy(model.state_dict())
            checkpoint_saved = True
        else:
            stale += 1
        print(
            f"[epoch {_epoch + 1:02d}/{epochs:02d}] model=student "
            f"train_loss={epoch_loss_sum / epoch_example_count:.6f} "
            f"val_macro_auprc={val_ap:.6f} best_val_macro_auprc={best_ap:.6f} "
            f"patience={stale}/{config['patience']} "
            f"checkpoint={'saved' if checkpoint_saved else 'kept'}",
            flush=True,
        )
        if stale >= int(config["patience"]):
            print(
                f"[early-stop] model=student epoch={_epoch + 1} best_epoch={best_epoch}",
                flush=True,
            )
            break
    if best_state is None:
        raise RuntimeError("no validation-selected Student state was produced")
    model.load_state_dict(best_state)
    print(
        f"[selected] model=student best_epoch={best_epoch} "
        f"best_val_macro_auprc={best_ap:.6f}",
        flush=True,
    )
    return _require_defined_macro_ap(best_ap), best_state


def _train_teacher_stage(
    model: nn.Module,
    loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    config: Mapping[str, Any],
    hierarchy_loss: HierarchyLoss,
    stage: str,
    checkpoint_path: Path,
) -> float:
    if not hasattr(model, "set_training_stage"):
        raise TypeError("teacher model must expose staged training controls")
    model.set_training_stage(stage)
    parameters = model.baseline_parameters() if stage == "baseline" else model.auxiliary_parameters()
    optimizer = build_optimizer(model, config, parameters=parameters)
    best_ap = -float("inf")
    best_epoch = 0
    stale = 0
    epochs = int(config["epochs"])
    print(
        f"[training] model=teacher stage={stage} epochs={epochs} "
        f"patience={config['patience']} batches_per_epoch={len(loader)}",
        flush=True,
    )
    for _epoch in range(epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_example_count = 0
        for batch in loader:
            drug_a, drug_b, organ_targets, specific_targets = _teacher_or_student_batch(
                batch, device, False
            )
            optimizer.zero_grad()
            output = model(drug_a, drug_b)
            loss = _teacher_supervised_loss(
                output,
                organ_targets,
                specific_targets,
                hierarchy_loss,
                float(config["hierarchy_weight"]),
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite teacher {stage} training loss")
            loss.backward()
            optimizer.step()
            batch_example_count = int(specific_targets.shape[0])
            epoch_loss_sum += float(loss.detach().cpu()) * batch_example_count
            epoch_example_count += batch_example_count
        if epoch_example_count == 0:
            raise RuntimeError(f"no teacher {stage} training examples were processed")
        val_logits, val_targets = _predict(model, validation_loader, device, False)
        val_ap = _require_defined_macro_ap(_macro_ap(val_logits, val_targets))
        checkpoint_saved = False
        if val_ap > best_ap:
            best_ap = val_ap
            best_epoch = _epoch + 1
            stale = 0
            torch.save(model.state_dict(), checkpoint_path)
            checkpoint_saved = True
        else:
            stale += 1
        print(
            f"[epoch {_epoch + 1:02d}/{epochs:02d}] model=teacher stage={stage} "
            f"train_loss={epoch_loss_sum / epoch_example_count:.6f} "
            f"val_macro_auprc={val_ap:.6f} best_val_macro_auprc={best_ap:.6f} "
            f"patience={stale}/{config['patience']} "
            f"checkpoint={'saved' if checkpoint_saved else 'kept'}",
            flush=True,
        )
        if stale >= int(config["patience"]):
            print(
                f"[early-stop] model=teacher stage={stage} epoch={_epoch + 1} "
                f"best_epoch={best_epoch}",
                flush=True,
            )
            break
    if not checkpoint_path.is_file():
        raise RuntimeError(f"no validation-selected {stage} teacher checkpoint was produced")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.set_training_stage(stage)
    print(
        f"[selected] model=teacher stage={stage} best_epoch={best_epoch} "
        f"best_val_macro_auprc={best_ap:.6f}",
        flush=True,
    )
    return _require_defined_macro_ap(best_ap)


def _build_teacher_selection(
    variant_metrics: Mapping[str, Mapping[str, float]],
    diagnostics: Mapping[str, Any],
    promotion_min_delta: float,
) -> tuple[dict[str, Any], str]:
    baseline_ap = _require_defined_macro_ap(variant_metrics["baseline"]["macro_ap"])
    combined_metrics = variant_metrics.get("combined")
    fused_ap = (
        _require_defined_macro_ap(combined_metrics["macro_ap"])
        if combined_metrics is not None
        else baseline_ap
    )
    selected_mode = _select_teacher_promotion(baseline_ap, fused_ap, promotion_min_delta)
    selected_variant = "combined" if selected_mode == "fused" else "baseline"
    selection = {
        "baseline": {"macro_ap": baseline_ap},
        "per_modality": {
            name: dict(values)
            for name, values in variant_metrics.items()
            if name not in {"baseline", "combined"}
        },
        "combined": {
            "macro_ap": fused_ap,
            "skipped": combined_metrics is None,
        },
        "selected": {
            "mode": selected_mode,
            "macro_ap": _require_defined_macro_ap(
                variant_metrics[selected_variant]["macro_ap"]
            ),
        },
        "promotion_min_delta": float(promotion_min_delta),
        "diagnostics": dict(diagnostics),
    }
    return selection, selected_mode


def _train_teacher_two_stage(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    config: Mapping[str, Any],
    hierarchy_loss: HierarchyLoss,
    run_dir: Path,
) -> dict[str, Any]:
    baseline_checkpoint = run_dir / "checkpoint_baseline.pt"
    fused_checkpoint = run_dir / "checkpoint_fused.pt"
    model.unfreeze_baseline()
    baseline_ap = _train_teacher_stage(
        model, train_loader, validation_loader, device, config, hierarchy_loss,
        "baseline", baseline_checkpoint,
    )
    model.load_state_dict(torch.load(baseline_checkpoint, map_location=device, weights_only=True))
    model.set_training_stage("baseline")
    if not getattr(model, "enabled_modalities", ()):
        variant_predictions, diagnostics = _predict_teacher_variants(
            model, validation_loader, device
        )
        variant_metrics = {
            name: {"macro_ap": _require_defined_macro_ap(_macro_ap(logits, targets))}
            for name, (logits, targets) in variant_predictions.items()
        }
        selection, _ = _build_teacher_selection(
            variant_metrics,
            diagnostics,
            float(config.get("promotion_min_delta", 0.002)),
        )
        _atomic_write_json(run_dir / "teacher_validation_selection.json", selection)
        return selection
    model.freeze_baseline()
    fused_ap = _train_teacher_stage(
        model, train_loader, validation_loader, device, config, hierarchy_loss,
        "fused", fused_checkpoint,
    )
    model.load_state_dict(torch.load(fused_checkpoint, map_location=device, weights_only=True))
    model.set_training_stage("fused")
    variant_predictions, diagnostics = _predict_teacher_variants(model, validation_loader, device)
    variant_metrics = {
        name: {"macro_ap": _require_defined_macro_ap(_macro_ap(logits, targets))}
        for name, (logits, targets) in variant_predictions.items()
    }
    # Stage checkpoints are selected independently; use the actual post-load
    # validation variants for the promotion decision and reported metrics.
    selection, selected_mode = _build_teacher_selection(
        variant_metrics,
        diagnostics,
        float(config.get("promotion_min_delta", 0.002)),
    )
    selected_checkpoint = baseline_checkpoint if selected_mode == "baseline" else fused_checkpoint
    model.load_state_dict(torch.load(selected_checkpoint, map_location=device, weights_only=True))
    model.set_training_stage(selected_mode)
    _atomic_write_json(run_dir / "teacher_validation_selection.json", selection)
    return selection


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
    device = choose_device()
    model = plan.model.to(device)
    student_mode = plan.config["model_type"] == "distilled_pair_student"
    run_model_id = _derive_run_model_id(plan.config)
    run_id = f"{run_model_id}__{plan.manifest.get('scenario')}__seed_{plan.manifest['seed']}__{plan.manifest['manifest_hash']}"
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
            "run_model_id": run_model_id,
            "benchmark_id": plan.manifest["benchmark_id"],
            "scenario": plan.manifest["scenario"],
            "seed": int(plan.manifest["seed"]),
            "manifest_hash": plan.manifest["manifest_hash"],
            "train_prevalences": train_prevalences,
            "source_hashes": plan.source_hashes,
            "labels_artifact_sha256": plan.source_hashes["labels_artifact"],
            "labels_order_sha256": plan.source_hashes["labels_order"],
            "feature_artifact_sha256": plan.source_hashes.get("feature_artifact"),
            "device": str(device),
        }
    )
    if not student_mode:
        resolved.update(
            {
                "enabled_modalities": list(model.enabled_modalities),
            }
        )
    (run_dir / "config.resolved.json").write_text(json.dumps(resolved, indent=2))
    train_generator = torch.Generator()
    train_generator.manual_seed(int(plan.manifest["seed"]))
    train_loader = DataLoader(
        plan.train_dataset,
        batch_size=int(plan.config["batch_size"]),
        shuffle=not student_mode,
        generator=train_generator,
    )
    validation_loader = DataLoader(plan.validation_dataset, batch_size=int(plan.config["batch_size"]))
    teacher_train_loader = (
        DataLoader(plan.teacher_train_dataset, batch_size=int(plan.config["batch_size"]))
        if student_mode
        else None
    )
    teacher_validation_loader = (
        DataLoader(plan.teacher_validation_dataset, batch_size=int(plan.config["batch_size"]))
        if student_mode
        else None
    )
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
    teacher_selection: dict[str, Any] | None = None
    if student_mode:
        if (
            teacher_model is None
            or teacher_train_loader is None
            or teacher_validation_loader is None
        ):
            raise RuntimeError("student training requires optimizer, teacher model, and teacher data")
        initial_student_state = copy.deepcopy(model.state_dict())
        candidate_results: list[dict[str, Any]] = []
        candidate_states: dict[tuple[float, float], dict[str, torch.Tensor]] = {}
        for distillation_weight, temperature in _candidate_grid(plan.config):
            validation_ap, candidate_state = _train_student_candidate(
                model,
                initial_student_state,
                train_loader,
                teacher_train_loader,
                validation_loader,
                teacher_model,
                device,
                plan.config,
                hierarchy_loss,
                distillation_weight,
                temperature,
            )
            candidate = {
                "distillation_weight": distillation_weight,
                "temperature": temperature,
                "distillation_temperature": temperature,
                "validation_macro_ap": validation_ap,
            }
            candidate_results.append(candidate)
            candidate_states[(distillation_weight, temperature)] = candidate_state
        selected_candidate = _select_distillation_candidate(candidate_results)
        selected_key = (
            float(selected_candidate["distillation_weight"]),
            float(selected_candidate["temperature"]),
        )
        model.load_state_dict(candidate_states[selected_key])
        student_val_logits, student_val_targets = _predict(model, validation_loader, device, True)
        student_val_ap = _require_defined_macro_ap(_macro_ap(student_val_logits, student_val_targets))
        teacher_val_logits, teacher_val_targets = _predict(
            teacher_model, teacher_validation_loader, device, False
        )
        teacher_val_ap = _require_defined_macro_ap(_macro_ap(teacher_val_logits, teacher_val_targets))
        max_drop = float(plan.config.get("student_max_validation_ap_drop", 0.005))
        _validate_student_teacher_ap(student_val_ap, teacher_val_ap, max_drop)
        student_selection = {
            "candidates": candidate_results,
            "results": candidate_results,
            "selected": selected_candidate,
            "selected_student_validation_macro_ap": student_val_ap,
            "selected_teacher_validation_macro_ap": teacher_val_ap,
            "student_max_validation_ap_drop": max_drop,
        }
        _atomic_write_json(run_dir / "student_distillation_selection.json", student_selection)
        resolved.update(
            {
                "student_distillation_selection_path": "student_distillation_selection.json",
                "student_distillation_selection_sha256": _sha256(
                    run_dir / "student_distillation_selection.json"
                ),
                "student_selected_distillation_weight": selected_candidate["distillation_weight"],
                "student_selected_temperature": selected_candidate["temperature"],
                "student_validation_macro_ap": student_val_ap,
                "teacher_validation_macro_ap": teacher_val_ap,
            }
        )
        (run_dir / "config.resolved.json").write_text(json.dumps(resolved, indent=2))
        torch.save(model.state_dict(), run_dir / "checkpoint_best.pt")
    else:
        teacher_selection = _train_teacher_two_stage(
            model,
            train_loader,
            validation_loader,
            device,
            plan.config,
            hierarchy_loss,
            run_dir,
        )
        torch.save(model.state_dict(), run_dir / "checkpoint_best.pt")
        resolved.update(
            {
                "teacher_selected_mode": teacher_selection["selected"]["mode"],
                "teacher_validation_selection_path": "teacher_validation_selection.json",
                "teacher_validation_selection_sha256": _sha256(
                    run_dir / "teacher_validation_selection.json"
                ),
                "promotion_min_delta": teacher_selection["promotion_min_delta"],
            }
        )
        (run_dir / "config.resolved.json").write_text(json.dumps(resolved, indent=2))
    trainer.model_selected()
    model_name = "student" if student_mode else "teacher"
    print("[phase] validation calibration and threshold selection", flush=True)
    val_logits, val_targets = _predict(model, validation_loader, device, student_mode)
    labels = _label_names(plan.validation_dataset)
    val_logits_frame = pd.DataFrame(val_logits, columns=labels)
    val_targets_frame = pd.DataFrame(val_targets, columns=labels)
    calibrated, temperatures = calibrate_validation_logits({"specific": val_logits_frame}, {"specific": val_targets_frame})
    print(
        f"[calibration] model={model_name} temperature={temperatures['specific']:.6f}",
        flush=True,
    )
    threshold_started = time.monotonic()
    thresholds = select_thresholds(val_targets_frame, calibrated["specific"])
    print(
        f"[thresholds] model={model_name} labels={len(thresholds)} "
        f"elapsed_seconds={time.monotonic() - threshold_started:.3f}",
        flush=True,
    )
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
    trainer.complete(metrics, required_artifacts=_required_run_artifacts(student_mode=student_mode))
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

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from run_precomputed_experiment import preflight_experiment

from .schemas import OrganPrediction, PredictionResponse, SpecificPrediction
from .settings import Settings


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _tensorize(values: Mapping[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for key, value in values.items():
        tensor = torch.from_numpy(np.asarray(value)).unsqueeze(0)
        tensors[key] = tensor.to(device=device)
    return tensors


class UnsupportedDrugError(ValueError):
    def __init__(self, drug_ids: list[str]):
        super().__init__(f"unsupported drug identifiers: {', '.join(drug_ids)}")
        self.drug_ids = drug_ids


class ModelService:
    """Loads the frozen Teacher once and serves thread-safe pair predictions."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = torch.device(settings.device)
        self._lock = threading.Lock()
        self.ready = False

    def load(self) -> None:
        for path in (
            self.settings.experiment_path,
            self.settings.manifest_path,
            self.settings.checkpoint_path,
            self.settings.selection_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("POLYPHARMACY_DEVICE requests CUDA but CUDA is unavailable")
        if self.device.type == "cpu":
            torch.set_num_threads(self.settings.cpu_threads)

        checkpoint_hash = _sha256(self.settings.checkpoint_path)
        selection_hash = _sha256(self.settings.selection_path)
        if self.settings.checkpoint_sha256 and checkpoint_hash != self.settings.checkpoint_sha256:
            raise ValueError("checkpoint SHA-256 mismatch")
        if self.settings.selection_sha256 and selection_hash != self.settings.selection_sha256:
            raise ValueError("selection SHA-256 mismatch")

        plan = preflight_experiment(
            self.settings.experiment_path,
            self.settings.manifest_path,
        )
        selection = _read_object(self.settings.selection_path, "selection")
        selected = selection.get("selected")
        if not isinstance(selected, dict) or selected.get("mode") not in {"baseline", "fused"}:
            raise ValueError("selection must contain selected.mode=baseline|fused")
        state = torch.load(self.settings.checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint must contain a state-dict mapping")

        fused_model = plan.model
        fused_model.load_state_dict(state)
        fused_model.set_training_stage(str(selected["mode"]))
        fused_model.to(device=self.device).eval()
        for parameter in fused_model.parameters():
            parameter.requires_grad_(False)

        baseline_model = copy.deepcopy(fused_model)
        baseline_model.set_training_stage("baseline")
        baseline_model.eval()

        labels = list(plan.validation_dataset.labels)
        feature_artifact = plan.feature_artifact
        if feature_artifact is None:
            raise ValueError("Teacher preflight did not provide a feature artifact")

        self.plan = plan
        self.feature_artifact = feature_artifact
        self.fused_model = fused_model
        self.baseline_model = baseline_model
        self.labels = labels
        self.organs = list(plan.hierarchy.organ_order)
        self.drug_ids = set(map(str, feature_artifact.drug_ids))
        self.selected_mode = str(selected["mode"])
        self.validation_macro_auprc = (
            float(selected["macro_ap"]) if selected.get("macro_ap") is not None else None
        )
        self.checkpoint_sha256 = checkpoint_hash
        self.selection_sha256 = selection_hash
        self.ready = True

    def _lookup(self, drug_id: str) -> Mapping[str, np.ndarray]:
        return self.feature_artifact.lookup(drug_id)

    def predict(self, drug_a: str, drug_b: str, top_k: int) -> PredictionResponse:
        if not self.ready:
            raise RuntimeError("model service is not ready")
        top_k = min(top_k, self.settings.max_top_k)
        unknown = [drug for drug in (drug_a, drug_b) if drug not in self.drug_ids]
        if unknown:
            raise UnsupportedDrugError(sorted(set(unknown)))

        first, second = sorted((drug_a, drug_b))
        features_a = self._lookup(first)
        features_b = self._lookup(second)
        morgan_ready = bool(features_a["morgan_available"]) and bool(
            features_b["morgan_available"]
        )
        if not morgan_ready:
            raise UnsupportedDrugError([drug for drug in (first, second) if not bool(self._lookup(drug)["morgan_available"])])
        mpnn_ready = bool(features_a["mpnn_available"]) and bool(features_b["mpnn_available"])
        use_fused = self.selected_mode == "fused" and mpnn_ready
        model = self.fused_model if use_fused else self.baseline_model
        mode = "fused" if use_fused else "baseline"
        warnings: list[str] = []
        confidence = "standard"
        if mode == "baseline" and self.selected_mode == "fused":
            confidence = "reduced"
            warnings.append("MPNN features are unavailable; Morgan baseline fallback was used.")

        input_a = _tensorize(features_a, self.device)
        input_b = _tensorize(features_b, self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter_ns()
        with self._lock, torch.inference_mode():
            output = model(input_a, input_b)
            specific = torch.sigmoid(output.specific_logits[0])
            organ = torch.sigmoid(output.organ_logits[0])
            specific_values, specific_indices = torch.topk(
                specific, k=min(top_k, specific.numel())
            )
            organ_values, organ_indices = torch.topk(organ, k=min(top_k, organ.numel()))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_ms = (time.perf_counter_ns() - started) / 1_000_000.0

        specific_items = []
        for value, index in zip(specific_values.cpu().tolist(), specific_indices.cpu().tolist()):
            label = self.labels[index]
            specific_items.append(
                SpecificPrediction(
                    label=str(label.get("name", label.get("label", label.get("cui")))),
                    cui=str(label["cui"]),
                    probability=float(value),
                )
            )
        organ_items = [
            OrganPrediction(organ=str(self.organs[index]), probability=float(value))
            for value, index in zip(organ_values.cpu().tolist(), organ_indices.cpu().tolist())
        ]
        pair_id = hashlib.sha256(f"{first}|{second}".encode("utf-8")).hexdigest()[:24]
        return PredictionResponse(
            pair_id=pair_id,
            drug_a=first,
            drug_b=second,
            model="warm_morgan_mpnn_teacher",
            mode=mode,
            confidence=confidence,
            warnings=warnings,
            level_1_organs=organ_items,
            level_2_side_effects=specific_items,
            inference_ms=inference_ms,
        )

"""Single-pair CPU inference over a validated cached-token artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.features.cached_token_artifact import CachedTokenArtifact
from src.models.factory import DualHeadOutput


def _vector(
    value: Any,
    size: int,
    name: str,
    default: float,
    *,
    minimum: float = 0.0,
) -> torch.Tensor:
    if value is None:
        values = [default] * size
    elif isinstance(value, (float, int)) and not isinstance(value, bool):
        values = [float(value)] * size
    else:
        values = list(value)
    if len(values) != size:
        raise ValueError(f"{name} must contain exactly {size} values")
    tensor = torch.tensor(values, dtype=torch.float32)
    if not torch.isfinite(tensor).all() or (tensor < minimum).any():
        qualifier = "positive" if minimum > 0 else "non-negative"
        raise ValueError(f"{name} must contain finite {qualifier} values")
    return tensor


class CPUPairPredictor:
    """Validated CPU wrapper for one unordered drug pair."""

    def __init__(
        self,
        student: torch.nn.Module,
        cache: CachedTokenArtifact | str | Path,
        *,
        label_names: list[str] | tuple[str, ...],
        organ_label_names: list[str] | tuple[str, ...] | None = None,
        temperatures: Mapping[str, Any] | None = None,
        thresholds: Mapping[str, Any] | None = None,
        top_k: int = 5,
    ) -> None:
        if not isinstance(student, torch.nn.Module):
            raise TypeError("student must be a torch module")
        if student.training:
            raise ValueError("student must be in eval() mode for CPU inference")
        if any(parameter.device.type != "cpu" for parameter in student.parameters()):
            raise ValueError("student parameters must be on CPU")
        if isinstance(cache, (str, Path)):
            cache = CachedTokenArtifact.load(cache)
        if not isinstance(cache, CachedTokenArtifact):
            raise TypeError("cache must be a validated CachedTokenArtifact or path")
        if (
            getattr(student, "token_count", None) != cache.token_count
            or getattr(student, "token_dim", None) != cache.token_dim
        ):
            raise ValueError("cached-token schema does not match the student token_count/token_dim")
        labels = tuple(str(label).strip() for label in label_names)
        if not labels or any(not label for label in labels) or len(set(labels)) != len(labels):
            raise ValueError("label_names must be unique non-empty names")
        organ_labels = tuple(organ_label_names or [f"organ_{i}" for i in range(student.num_organ)])
        if len(organ_labels) != student.num_organ:
            raise ValueError("organ_label_names must match the student's organ head")
        if len(labels) != student.num_specific:
            raise ValueError("label_names must match the student's specific head")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        self.student = student
        self.cache = cache
        self.label_names = labels
        self.organ_label_names = organ_labels
        temperature_config = temperatures or {}
        threshold_config = thresholds or {}
        self.organ_temperatures = _vector(
            temperature_config.get("organ"),
            student.num_organ,
            "organ temperatures",
            1.0,
            minimum=1e-6,
        )
        self.specific_temperatures = _vector(
            temperature_config.get("specific"),
            student.num_specific,
            "specific temperatures",
            1.0,
            minimum=1e-6,
        )
        self.organ_thresholds = _vector(
            threshold_config.get("organ"), student.num_organ, "organ thresholds", 0.5
        )
        self.specific_thresholds = _vector(
            threshold_config.get("specific"), student.num_specific, "specific thresholds", 0.5
        )
        if (self.organ_thresholds > 1).any() or (self.specific_thresholds > 1).any():
            raise ValueError("thresholds must be between zero and one")
        self.top_k = min(top_k, student.num_specific)

    @staticmethod
    def _read_json(path: str | Path, name: str) -> dict[str, Any]:
        try:
            value = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"unable to read {name} artifact {path}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{name} artifact must contain a JSON object")
        return value

    @staticmethod
    def _threshold_records(
        records: Mapping[str, Any], labels: tuple[str, ...], name: str
    ) -> list[float]:
        if set(records) != set(labels):
            raise ValueError(f"{name} threshold labels must exactly match model labels")
        values: list[float] = []
        for label in labels:
            record = records[label]
            if not isinstance(record, Mapping) or "threshold" not in record:
                raise ValueError(f"{name} threshold record for {label} must contain threshold")
            values.append(record["threshold"])
        return values

    @classmethod
    def from_artifacts(
        cls,
        student: torch.nn.Module,
        cache: CachedTokenArtifact | str | Path,
        *,
        label_names: list[str] | tuple[str, ...],
        calibration_path: str | Path,
        thresholds_path: str | Path,
        organ_label_names: list[str] | tuple[str, ...] | None = None,
        top_k: int = 5,
    ) -> "CPUPairPredictor":
        """Construct a predictor from the runner's calibration artifacts."""

        calibration = cls._read_json(calibration_path, "calibration")
        temperatures = calibration.get("temperatures")
        if not isinstance(temperatures, Mapping) or "specific" not in temperatures:
            raise ValueError("calibration artifact must contain temperatures.specific")
        raw_thresholds = cls._read_json(thresholds_path, "thresholds")
        if "specific" in raw_thresholds:
            specific_thresholds = raw_thresholds["specific"]
            if not isinstance(specific_thresholds, Mapping):
                raise ValueError("thresholds.specific must be label-keyed records")
        else:
            specific_thresholds = raw_thresholds
        labels = tuple(str(label).strip() for label in label_names)
        specific_values = cls._threshold_records(specific_thresholds, labels, "specific")
        predictor_temperatures: dict[str, Any] = {"specific": temperatures["specific"]}
        predictor_thresholds: dict[str, Any] = {"specific": specific_values}
        organs = tuple(organ_label_names or ())
        if "organ" in temperatures:
            predictor_temperatures["organ"] = temperatures["organ"]
        if "organ" in raw_thresholds:
            if not organs:
                raise ValueError("organ thresholds require organ_label_names")
            organ_records = raw_thresholds["organ"]
            if not isinstance(organ_records, Mapping):
                raise ValueError("thresholds.organ must be label-keyed records")
            predictor_thresholds["organ"] = cls._threshold_records(organ_records, organs, "organ")
        return cls(
            student,
            cache,
            label_names=labels,
            organ_label_names=organ_label_names,
            temperatures=predictor_temperatures,
            thresholds=predictor_thresholds,
            top_k=top_k,
        )

    @staticmethod
    def _top_k(probabilities: np.ndarray, labels: tuple[str, ...], count: int) -> list[dict[str, Any]]:
        order = sorted(range(len(labels)), key=lambda index: (-float(probabilities[index]), index))
        return [
            {
                "index": index,
                "label": labels[index],
                "probability": float(probabilities[index]),
            }
            for index in order[:count]
        ]

    def predict_pair(self, drug_a: str, drug_b: str) -> dict[str, Any]:
        tokens_a, availability_a, tokens_b, availability_b = self.cache.lookup_pair(drug_a, drug_b)
        device = torch.device("cpu")
        with torch.inference_mode():
            output = self.student(
                torch.from_numpy(tokens_a).unsqueeze(0).to(device),
                torch.from_numpy(tokens_b).unsqueeze(0).to(device),
                torch.from_numpy(availability_a).unsqueeze(0).to(device),
                torch.from_numpy(availability_b).unsqueeze(0).to(device),
            )
        if not isinstance(output, DualHeadOutput):
            raise ValueError("student must emit DualHeadOutput")
        if not torch.isfinite(output.organ_logits).all() or not torch.isfinite(output.specific_logits).all():
            raise ValueError("student emitted non-finite logits")
        organ_logits = output.organ_logits[0].cpu() / self.organ_temperatures
        specific_logits = output.specific_logits[0].cpu() / self.specific_temperatures
        organ_probabilities = torch.sigmoid(organ_logits).numpy()
        specific_probabilities = torch.sigmoid(specific_logits).numpy()
        specific_predictions = specific_probabilities >= self.specific_thresholds.numpy()
        organ_predictions = organ_probabilities >= self.organ_thresholds.numpy()
        top_k = self._top_k(specific_probabilities, self.label_names, self.top_k)
        for item in top_k:
            item["predicted"] = bool(specific_predictions[item["index"]])
        return {
            "drug_a": drug_a,
            "drug_b": drug_b,
            "organ_logits": output.organ_logits[0].cpu().numpy(),
            "specific_logits": output.specific_logits[0].cpu().numpy(),
            "organ_probabilities": organ_probabilities,
            "specific_probabilities": specific_probabilities,
            "organ_predictions": organ_predictions,
            "specific_predictions": specific_predictions,
            "top_k": top_k,
        }

    def predict_pair_json(self, drug_a: str, drug_b: str) -> dict[str, Any]:
        """Return the pair result using only JSON-native scalar/list values."""

        result = self.predict_pair(drug_a, drug_b)

        def convert(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (np.floating, np.integer, np.bool_)):
                return value.item()
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, list):
                return [convert(item) for item in value]
            return value

        return convert(result)

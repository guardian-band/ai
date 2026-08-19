"""Honest model/strategy dispatch for manifest-backed deterministic inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, NamedTuple

import torch
import torch.nn as nn
import yaml


class UnsupportedModelConfiguration(ValueError):
    """Raised when a declared model needs unavailable benchmark features."""


SUPPORTED_MODEL_TYPES = {
    "prevalence",
    "logistic",
    "symmetric_mlp",
    "morgan_graphsage_mlp",
    "multimodal_teacher",
    "distilled_pair_student",
}


def validate_model_type(config: Mapping[str, Any]) -> str:
    model_type = config.get("model_type") if isinstance(config, Mapping) else None
    if model_type == "unified":
        raise UnsupportedModelConfiguration(
            "model_type 'unified' is unavailable: PrimeKG graph inputs are not present in the manifest dataset"
        )
    if model_type == "advanced":
        raise UnsupportedModelConfiguration(
            "model_type 'advanced' is unavailable: SIDER, ChemBERTa, and PrimeKG inputs are not present"
        )
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise UnsupportedModelConfiguration(f"Unknown or missing model_type: {model_type!r}")
    return model_type


def validate_precomputed_model_runner(config: Mapping[str, Any]) -> None:
    """Reject new architectures before the manifest runner creates artifacts.

    ``ManifestPolypharmacyDataset`` emits one legacy feature tensor per drug;
    it cannot safely coerce that tensor into multimodal teacher inputs or
    cached student tokens.  An explicit precomputed-token runner must be
    connected before these model types are trained through this entry point.
    """

    model_type = config.get("model_type") if isinstance(config, Mapping) else None
    if model_type not in {"multimodal_teacher", "distilled_pair_student"}:
        return
    artifact_key = (
        "cached_token_artifact_path"
        if model_type == "distilled_pair_student"
        else "feature_artifact_path"
    )
    if not config.get(artifact_key):
        raise UnsupportedModelConfiguration(
            f"model_type '{model_type}' requires a validated precomputed artifact via "
            f"{artifact_key}; ManifestPolypharmacyDataset cannot supply its token schema"
        )
    raise UnsupportedModelConfiguration(
        f"model_type '{model_type}' is not connected to run_experiment: "
        "use an explicit precomputed-token training entry point"
    )


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate one experiment YAML mapping."""

    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Unable to read experiment config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Experiment config must be a YAML mapping")
    model_type = raw.get("model_type")
    if not isinstance(model_type, str) or not model_type.strip():
        raise ValueError("Experiment config must define model_type")
    resolved = dict(raw)
    resolved["model_type"] = model_type.strip().lower()
    if resolved["model_type"] not in SUPPORTED_MODEL_TYPES | {"unified", "advanced"}:
        raise UnsupportedModelConfiguration(
            f"Unknown model_type '{model_type}'. Expected a supported model type."
        )
    return resolved


def _masked_mean(tokens: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
    if tokens.ndim == 2:
        tokens = tokens.unsqueeze(0)
    if padding_mask is None:
        return tokens.mean(dim=1)
    if padding_mask.ndim == 1:
        padding_mask = padding_mask.unsqueeze(0)
    valid = (~padding_mask).to(dtype=tokens.dtype).unsqueeze(-1)
    return (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)


def symmetric_pair_representation(
    drug_a_tokens: torch.Tensor,
    drug_b_tokens: torch.Tensor,
    mask_a: torch.Tensor | None = None,
    mask_b: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the order-invariant ``[sum, abs-diff, product]`` representation."""

    drug_a = _masked_mean(drug_a_tokens, mask_a)
    drug_b = _masked_mean(drug_b_tokens, mask_b)
    return torch.cat([drug_a + drug_b, torch.abs(drug_a - drug_b), drug_a * drug_b], dim=1)


class PrevalenceModel(nn.Module):
    """Fixed baseline that predicts the train-label prevalence for every pair."""

    is_fixed_strategy = True

    def __init__(self, num_labels: int):
        super().__init__()
        self.register_buffer("prevalence", torch.zeros(num_labels, dtype=torch.float32))
        self.register_buffer("_fitted", torch.tensor(False, dtype=torch.bool))

    def fit(self, train_labels: torch.Tensor) -> None:
        if train_labels.ndim != 2 or train_labels.shape[1] != self.prevalence.numel():
            raise ValueError("train labels must be a 2D tensor matching num_labels")
        if train_labels.shape[0] == 0:
            raise ValueError("cannot fit prevalence on an empty training split")
        self.prevalence.copy_(train_labels.to(self.prevalence.device).float().mean(dim=0))
        self._fitted.fill_(True)

    def predict_proba(self, batch_size: int) -> torch.Tensor:
        if not bool(self._fitted.item()):
            raise RuntimeError("PrevalenceModel.fit must be called before prediction")
        return self.prevalence.unsqueeze(0).expand(batch_size, -1)

    def forward(self, drug_a_tokens, drug_b_tokens, mask_a=None, mask_b=None):
        probabilities = self.predict_proba(drug_a_tokens.shape[0])
        return torch.logit(probabilities.clamp(1e-6, 1.0 - 1e-6))


class LogisticPairModel(nn.Module):
    """One-vs-rest linear classifier over a symmetric pair representation."""

    is_fixed_strategy = False

    def __init__(self, input_dim: int, num_labels: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim * 3, num_labels)

    def forward(self, drug_a_tokens, drug_b_tokens, mask_a=None, mask_b=None):
        features = symmetric_pair_representation(drug_a_tokens, drug_b_tokens, mask_a, mask_b)
        return self.classifier(features)


class SymmetricMLPModel(nn.Module):
    """Shared drug encoder followed by symmetric pair features and an MLP head."""

    is_fixed_strategy = False

    def __init__(self, input_dim: int, num_labels: int, config: Mapping[str, Any] | None = None):
        super().__init__()
        config = config or {}
        encoder_config = config.get("encoder") if isinstance(config.get("encoder"), dict) else {}
        decoder_config = config.get("decoder") if isinstance(config.get("decoder"), dict) else {}
        encoder_layers = encoder_config.get("layers", [128])
        decoder_layers = decoder_config.get("layers", [64])
        hidden_dim = int(encoder_layers[-1]) if encoder_layers else 128
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU())
        decoder_modules: list[nn.Module] = []
        current_dim = hidden_dim * 3
        for layer_dim in decoder_layers:
            layer_dim = int(layer_dim)
            decoder_modules.extend([nn.Linear(current_dim, layer_dim), nn.GELU()])
            current_dim = layer_dim
        decoder_modules.append(nn.Linear(current_dim, num_labels))
        self.decoder = nn.Sequential(*decoder_modules)

    def forward(self, drug_a_tokens, drug_b_tokens, mask_a=None, mask_b=None):
        drug_a = self.encoder(_masked_mean(drug_a_tokens, mask_a))
        drug_b = self.encoder(_masked_mean(drug_b_tokens, mask_b))
        features = torch.cat([drug_a + drug_b, torch.abs(drug_a - drug_b), drug_a * drug_b], dim=1)
        return self.decoder(features)


class DualHeadOutput(NamedTuple):
    """Named logits emitted by the reusable hierarchy decoder."""

    organ_logits: torch.Tensor
    specific_logits: torch.Tensor


class SymmetricDualHeadModel(nn.Module):
    """Order-invariant shared drug encoder with typed organ/specific heads."""

    def __init__(
        self,
        input_dim: int,
        num_organ: int,
        num_specific: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        if num_organ <= 0 or num_specific <= 0:
            raise ValueError("num_organ and num_specific must be positive")
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU())
        pair_dim = hidden_dim * 3
        self.organ_head = nn.Linear(pair_dim, num_organ)
        self.specific_head = nn.Linear(pair_dim, num_specific)

    def forward(self, drug_a_tokens, drug_b_tokens, mask_a=None, mask_b=None) -> DualHeadOutput:
        drug_a = self.encoder(_masked_mean(drug_a_tokens, mask_a))
        drug_b = self.encoder(_masked_mean(drug_b_tokens, mask_b))
        pair = torch.cat(
            [drug_a + drug_b, torch.abs(drug_a - drug_b), drug_a * drug_b],
            dim=1,
        )
        return DualHeadOutput(self.organ_head(pair), self.specific_head(pair))


def create_model(config: Mapping[str, Any], num_labels: int, input_dim: int = 768) -> nn.Module:
    """Instantiate exactly the model declared by ``config``."""

    model_type = validate_model_type(config)
    if model_type == "prevalence":
        return PrevalenceModel(num_labels=num_labels)
    if model_type == "logistic":
        return LogisticPairModel(input_dim=input_dim, num_labels=num_labels)
    if model_type in {"symmetric_mlp", "morgan_graphsage_mlp"}:
        return SymmetricMLPModel(input_dim=input_dim, num_labels=num_labels, config=config)
    if model_type == "multimodal_teacher":
        from src.models.multimodal_teacher_student import MultiModalTeacher

        num_specific = config.get("num_specific", num_labels)
        if num_specific != num_labels:
            raise ValueError("multimodal_teacher num_specific must match num_labels")
        required = (
            "molformer_dim",
            "mpnn_dim",
            "kg_dim",
            "molformer_token_count",
            "mpnn_token_count",
            "kg_token_count",
            "hidden_dim",
            "num_organ",
        )
        missing = [key for key in required if key not in config]
        if missing:
            raise UnsupportedModelConfiguration(
                f"multimodal_teacher requires explicit dimensions: {', '.join(missing)}"
            )
        return MultiModalTeacher(
            morgan_dim=config.get("morgan_dim", input_dim),
            molformer_dim=config["molformer_dim"],
            mpnn_dim=config["mpnn_dim"],
            kg_dim=config["kg_dim"],
            molformer_token_count=config["molformer_token_count"],
            mpnn_token_count=config["mpnn_token_count"],
            kg_token_count=config["kg_token_count"],
            hidden_dim=config["hidden_dim"],
            num_organ=config["num_organ"],
            num_specific=num_specific,
            num_heads=config.get("num_heads", 4),
            dropout=config.get("dropout", 0.0),
            cache_token_count=config.get("cache_token_count", 8),
            cache_token_dim=config.get("cache_token_dim", 128),
        )
    if model_type == "distilled_pair_student":
        from src.models.multimodal_teacher_student import DistilledPairStudent

        num_specific = config.get("num_specific", num_labels)
        if num_specific != num_labels:
            raise ValueError("distilled_pair_student num_specific must match num_labels")
        required = ("token_dim", "token_count", "hidden_dim", "num_organ")
        missing = [key for key in required if key not in config]
        if missing:
            raise UnsupportedModelConfiguration(
                f"distilled_pair_student requires explicit dimensions: {', '.join(missing)}"
            )
        return DistilledPairStudent(
            token_dim=config["token_dim"],
            token_count=config["token_count"],
            hidden_dim=config["hidden_dim"],
            num_organ=config["num_organ"],
            num_specific=num_specific,
            num_heads=config.get("num_heads", 4),
            dropout=config.get("dropout", 0.0),
        )
    raise UnsupportedModelConfiguration(f"Unknown or missing model_type: {model_type!r}")

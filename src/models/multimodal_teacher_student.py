"""Offline multimodal teacher and CPU-safe distilled pair student.

The teacher consumes independently produced Morgan, MolFormer, molecular-MPNN,
and typed PrimeKG-HGT features. Both pair models use shared, symmetric pair
features so the public prediction is invariant to the order of the drugs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.factory import (
    DualHeadOutput,
    TEACHER_AUXILIARY_MODALITIES,
    validate_enabled_modalities,
)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_float(value: float, name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not torch.isfinite(torch.tensor(float(value))):
        raise ValueError(f"{name} must be finite")
    value = float(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _check_tensor(value: Any, name: str, ndim: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D torch tensor")
    if not value.is_floating_point():
        raise ValueError(f"{name} must use a floating-point dtype")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


def _availability(value: Any, batch_size: int, name: str) -> torch.Tensor:
    if value is None:
        return torch.ones(batch_size, dtype=torch.bool)
    if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
        raise ValueError(f"{name} must be a boolean tensor")
    if value.ndim != 1 or value.shape[0] != batch_size:
        raise ValueError(f"{name} must have shape [batch]")
    return value


def _padding_mask(value: Any, batch_size: int, token_count: int, name: str) -> torch.Tensor:
    if value is None:
        return torch.zeros(batch_size, token_count, dtype=torch.bool)
    if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
        raise ValueError(f"{name} must be a boolean tensor")
    if value.shape != (batch_size, token_count):
        raise ValueError(f"{name} must have shape [batch, {token_count}]")
    return value


class MultiModalTeacher(nn.Module):
    """Token-level multimodal teacher for offline training.

    Each drug mapping contains ``morgan`` [B, M], ``molformer_tokens``
    [B, Tf, Df], ``mpnn_tokens`` [B, Tp, Dp], and ``kg_tokens`` [B, Tk, Dk].
    Availability fields are boolean per-drug masks. Padding masks use ``True``
    for padding. The model does not load or infer any upstream encoder.
    """

    def __init__(
        self,
        *,
        morgan_dim: int,
        molformer_dim: int,
        mpnn_dim: int,
        kg_dim: int,
        molformer_token_count: int,
        mpnn_token_count: int,
        kg_token_count: int,
        hidden_dim: int,
        num_organ: int,
        num_specific: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        cache_token_count: int = 8,
        cache_token_dim: int = 128,
        modality_summary_token_count: int = 4,
        modality_dropout: float = 0.0,
        enabled_modalities: list[str] | tuple[str, ...] | None = None,
        modality_gate_init_logit: float = -4.0,
    ) -> None:
        super().__init__()
        morgan_dim = _positive_int(morgan_dim, "morgan_dim")
        molformer_dim = _positive_int(molformer_dim, "molformer_dim")
        mpnn_dim = _positive_int(mpnn_dim, "mpnn_dim")
        kg_dim = _positive_int(kg_dim, "kg_dim")
        molformer_token_count = _positive_int(
            molformer_token_count, "molformer_token_count"
        )
        mpnn_token_count = _positive_int(mpnn_token_count, "mpnn_token_count")
        kg_token_count = _positive_int(kg_token_count, "kg_token_count")
        hidden_dim = _positive_int(hidden_dim, "hidden_dim")
        num_organ = _positive_int(num_organ, "num_organ")
        num_specific = _positive_int(num_specific, "num_specific")
        num_heads = _positive_int(num_heads, "num_heads")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        dropout = _validate_float(dropout, "dropout")
        if dropout >= 1.0:
            raise ValueError("dropout must be less than 1")
        cache_token_count = _positive_int(cache_token_count, "cache_token_count")
        cache_token_dim = _positive_int(cache_token_dim, "cache_token_dim")
        modality_summary_token_count = _positive_int(
            modality_summary_token_count, "modality_summary_token_count"
        )
        modality_dropout = _validate_float(modality_dropout, "modality_dropout")
        if modality_dropout >= 1.0:
            raise ValueError("modality_dropout must be less than 1")
        modality_gate_init_logit = _validate_float(
            modality_gate_init_logit,
            "modality_gate_init_logit",
            minimum=-float("inf"),
        )

        self.morgan_dim = morgan_dim
        self.molformer_dim = molformer_dim
        self.mpnn_dim = mpnn_dim
        self.kg_dim = kg_dim
        self.molformer_token_count = molformer_token_count
        self.mpnn_token_count = mpnn_token_count
        self.kg_token_count = kg_token_count
        self.hidden_dim = hidden_dim
        self.num_organ = num_organ
        self.num_specific = num_specific
        self.cache_token_count = cache_token_count
        self.cache_token_dim = cache_token_dim
        self.modality_summary_token_count = modality_summary_token_count
        self.modality_dropout = modality_dropout
        self.enabled_modalities = validate_enabled_modalities(enabled_modalities)
        self.modality_gate_init_logit = modality_gate_init_logit
        self.training_stage = "fused"
        self.morgan_projection = nn.Sequential(
            nn.Linear(morgan_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        modality_dims = {
            "molformer": molformer_dim,
            "mpnn": mpnn_dim,
            "kg": kg_dim,
        }
        self.modality_projections = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(dimension, hidden_dim), nn.LayerNorm(hidden_dim))
                for name, dimension in modality_dims.items()
            }
        )
        self.modality_queries = nn.ParameterDict(
            {
                name: nn.Parameter(
                    torch.randn(modality_summary_token_count, hidden_dim) * 0.02
                )
                for name in modality_dims
            }
        )
        self.modality_resamplers = nn.ModuleDict(
            {
                name: nn.MultiheadAttention(
                    hidden_dim, num_heads, dropout=dropout, batch_first=True
                )
                for name in modality_dims
            }
        )
        self.modality_summary_norms = nn.ModuleDict(
            {name: nn.LayerNorm(hidden_dim) for name in modality_dims}
        )
        self.modality_missing_tokens = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(hidden_dim)) for name in modality_dims}
        )
        self.modality_gate_logits = nn.ParameterDict(
            {
                name: nn.Parameter(torch.full((2,), modality_gate_init_logit))
                for name in modality_dims
            }
        )
        pair_dim = hidden_dim * 3
        self.morgan_missing_token = nn.Parameter(torch.zeros(hidden_dim))
        self.morgan_baseline_organ_head = nn.Sequential(
            nn.LayerNorm(pair_dim), nn.Linear(pair_dim, num_organ)
        )
        self.morgan_baseline_specific_head = nn.Sequential(
            nn.LayerNorm(pair_dim), nn.Linear(pair_dim, num_specific)
        )
        self.modality_organ_heads = nn.ModuleDict(
            {
                name: nn.Sequential(nn.LayerNorm(pair_dim), nn.Linear(pair_dim, num_organ))
                for name in modality_dims
            }
        )
        self.modality_specific_heads = nn.ModuleDict(
            {
                name: nn.Sequential(nn.LayerNorm(pair_dim), nn.Linear(pair_dim, num_specific))
                for name in modality_dims
            }
        )
        self.cache_queries = nn.Parameter(torch.randn(cache_token_count, hidden_dim) * 0.02)
        self.cache_resampler = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cache_output_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, cache_token_dim)
        )
        self.cache_input_projection = nn.Linear(cache_token_dim, hidden_dim)
        self._last_diagnostics: dict[str, Any] = {
            "gates": {},
            "coverage": {},
        }
        for name in TEACHER_AUXILIARY_MODALITIES:
            if name in self.enabled_modalities:
                continue
            frozen_modules = (
                self.modality_projections[name],
                self.modality_resamplers[name],
                self.modality_summary_norms[name],
                self.modality_organ_heads[name],
                self.modality_specific_heads[name],
            )
            for module in frozen_modules:
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
            for parameter in (
                self.modality_queries[name],
                self.modality_missing_tokens[name],
                self.modality_gate_logits[name],
            ):
                parameter.requires_grad_(False)

    def _validate_drug(self, drug: Mapping[str, Any]) -> dict[str, Any]:
        required = {"morgan", "molformer_tokens", "mpnn_tokens", "kg_tokens"}
        if not isinstance(drug, Mapping) or not required.issubset(drug):
            raise ValueError(
                "each teacher drug input must contain morgan, molformer_tokens, "
                "mpnn_tokens, and kg_tokens"
            )
        morgan = _check_tensor(drug["morgan"], "morgan", 2)
        molformer = _check_tensor(drug["molformer_tokens"], "molformer_tokens", 3)
        mpnn = _check_tensor(drug["mpnn_tokens"], "mpnn_tokens", 3)
        kg = _check_tensor(drug["kg_tokens"], "kg_tokens", 3)
        batch_size = morgan.shape[0]
        if morgan.shape[1] != self.morgan_dim:
            raise ValueError(f"morgan must have shape [batch, {self.morgan_dim}]")
        if molformer.shape[0] != batch_size or molformer.shape[1:] != (
            self.molformer_token_count,
            self.molformer_dim,
        ):
            raise ValueError(
                "molformer_tokens must have shape "
                f"[batch, {self.molformer_token_count}, {self.molformer_dim}]"
            )
        if mpnn.shape[0] != batch_size or mpnn.shape[1:] != (
            self.mpnn_token_count,
            self.mpnn_dim,
        ):
            raise ValueError(
                f"mpnn_tokens must have shape [batch, {self.mpnn_token_count}, {self.mpnn_dim}]"
            )
        if kg.shape[0] != batch_size or kg.shape[1:] != (self.kg_token_count, self.kg_dim):
            raise ValueError(
                f"kg_tokens must have shape [batch, {self.kg_token_count}, {self.kg_dim}]"
            )
        morgan_available = _availability(
            drug.get("morgan_available"), batch_size, "morgan_available"
        ).to(device=morgan.device)
        molformer_available = _availability(
            drug.get("molformer_available"), batch_size, "molformer_available"
        ).to(device=morgan.device)
        mpnn_available = _availability(
            drug.get("mpnn_available"), batch_size, "mpnn_available"
        ).to(device=morgan.device)
        kg_available = _availability(drug.get("kg_available"), batch_size, "kg_available").to(
            device=morgan.device
        )
        molformer_mask = _padding_mask(
            drug.get("molformer_padding_mask"),
            batch_size,
            self.molformer_token_count,
            "molformer_padding_mask",
        ).to(device=morgan.device)
        mpnn_mask = _padding_mask(
            drug.get("mpnn_padding_mask"),
            batch_size,
            self.mpnn_token_count,
            "mpnn_padding_mask",
        ).to(device=morgan.device)
        kg_mask = _padding_mask(
            drug.get("kg_padding_mask"), batch_size, self.kg_token_count, "kg_padding_mask"
        ).to(device=morgan.device)

        return {
            "morgan": morgan,
            "morgan_available": morgan_available,
            "molformer": molformer,
            "molformer_available": molformer_available,
            "molformer_mask": molformer_mask,
            "mpnn": mpnn,
            "mpnn_available": mpnn_available,
            "mpnn_mask": mpnn_mask,
            "kg": kg,
            "kg_available": kg_available,
            "kg_mask": kg_mask,
        }

    def _resample_modality(
        self,
        name: str,
        values: torch.Tensor,
        availability: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.modality_projections[name](values)
        valid = availability[:, None] & ~padding_mask
        coverage = valid.sum(dim=1)
        missing_rows = coverage == 0
        if bool(missing_rows.any()):
            valid = valid.clone()
            projected = projected.clone()
            valid[missing_rows, 0] = True
            projected[missing_rows, 0] = self.modality_missing_tokens[name]
        projected = projected.masked_fill((~valid).unsqueeze(-1), 0.0)
        queries = self.modality_queries[name].unsqueeze(0).expand(values.shape[0], -1, -1)
        summary, _ = self.modality_resamplers[name](
            queries,
            projected,
            projected,
            key_padding_mask=~valid,
            need_weights=False,
        )
        summary = self.modality_summary_norms[name](summary)
        if not torch.isfinite(summary).all():
            raise RuntimeError(f"{name} modality resampler emitted non-finite values")
        return summary, coverage

    def _encode_modalities(
        self, drug: Mapping[str, Any]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        values = self._validate_drug(drug)
        morgan = self.morgan_projection(values["morgan"])
        morgan_missing = self.morgan_missing_token.view(1, -1)
        morgan_embedding = torch.where(
            values["morgan_available"][:, None], morgan, morgan_missing
        )
        summaries: dict[str, torch.Tensor] = {}
        coverage: dict[str, torch.Tensor] = {}
        for name in self.enabled_modalities:
            availability = values[f"{name}_available"]
            if self.training and self.modality_dropout > 0.0:
                keep = torch.rand(
                    availability.shape, device=availability.device
                ) >= self.modality_dropout
                availability = availability & keep
            summaries[name], coverage[name] = self._resample_modality(
                name,
                values[name],
                availability,
                values[f"{name}_mask"],
            )
        return morgan_embedding, summaries, coverage

    def encode_auxiliary_modalities(self, drug: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Return balanced summary tokens for the enabled auxiliary modalities."""

        _, summaries, _ = self._encode_modalities(drug)
        return summaries

    @staticmethod
    def _pair_features(drug_a: torch.Tensor, drug_b: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [drug_a + drug_b, torch.abs(drug_a - drug_b), drug_a * drug_b], dim=-1
        )

    def _baseline_from_embeddings(
        self, drug_a: torch.Tensor, drug_b: torch.Tensor
    ) -> DualHeadOutput:
        pair = self._pair_features(drug_a, drug_b)
        output = DualHeadOutput(
            self.morgan_baseline_organ_head(pair),
            self.morgan_baseline_specific_head(pair),
        )
        if not torch.isfinite(output.organ_logits).all() or not torch.isfinite(output.specific_logits).all():
            raise RuntimeError("Morgan baseline emitted non-finite logits")
        return output

    def morgan_baseline_logits(
        self, drug_a: Mapping[str, Any], drug_b: Mapping[str, Any]
    ) -> DualHeadOutput:
        values_a = self._validate_drug(drug_a)
        values_b = self._validate_drug(drug_b)
        embedding_a = self.morgan_projection(values_a["morgan"])
        embedding_b = self.morgan_projection(values_b["morgan"])
        embedding_a = torch.where(
            values_a["morgan_available"][:, None], embedding_a, self.morgan_missing_token
        )
        embedding_b = torch.where(
            values_b["morgan_available"][:, None], embedding_b, self.morgan_missing_token
        )
        return self._baseline_from_embeddings(embedding_a, embedding_b)

    def _cache_from_components(
        self,
        morgan: torch.Tensor,
        summaries: Mapping[str, torch.Tensor],
        modality_names: tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        names = self.enabled_modalities if modality_names is None else tuple(modality_names)
        if any(name not in summaries for name in names):
            raise ValueError("cache modality summaries are incomplete")
        projected_modalities = torch.cat(
            [morgan.unsqueeze(1)] + [summaries[name] for name in names],
            dim=1,
        )
        queries = self.cache_queries.unsqueeze(0).expand(projected_modalities.shape[0], -1, -1)
        latent_hidden, _ = self.cache_resampler(
            queries,
            projected_modalities,
            projected_modalities,
            need_weights=False,
        )
        latent_tokens = self.cache_output_projection(latent_hidden)
        if not torch.isfinite(latent_tokens).all():
            raise RuntimeError("teacher cache encoder emitted non-finite latent tokens")
        return latent_tokens

    def encode_drug_for_cache(self, drug: Mapping[str, Any]) -> torch.Tensor:
        """Return trainable fixed-shape latent tokens for one drug batch.

        This method intentionally shares all modality projections and the
        learned resampler used by the supervised pair path.  Calling it in
        ``eval()`` is deterministic; calling it in training mode preserves
        gradients into the cache encoder.
        """

        morgan, summaries, _ = self._encode_modalities(drug)
        return self._cache_from_components(morgan, summaries)

    def _forward_components(
        self, drug_a: Mapping[str, Any], drug_b: Mapping[str, Any]
    ) -> tuple[DualHeadOutput, dict[str, DualHeadOutput], DualHeadOutput]:
        morgan_a, summaries_a, coverage_a = self._encode_modalities(drug_a)
        morgan_b, summaries_b, coverage_b = self._encode_modalities(drug_b)
        if morgan_a.shape[0] != morgan_b.shape[0]:
            raise ValueError("drug A and drug B must have the same batch size")
        # Keep the cache encoder in the Teacher forward graph so cache artifacts
        # and supervised auxiliary paths share all frozen upstream modalities.
        baseline = self._baseline_from_embeddings(morgan_a, morgan_b)
        corrections: dict[str, DualHeadOutput] = {}
        organ_logits = baseline.organ_logits
        specific_logits = baseline.specific_logits
        gate_values: dict[str, torch.Tensor] = {}
        for name in self.enabled_modalities:
            # Each correction branch sees Morgan plus its own modality only.
            # The all-modality cache path remains in encode_drug_for_cache;
            # this isolated path prevents a single-modality validation variant
            # from silently carrying another modality's signal.
            cache_a = self._cache_from_components(morgan_a, summaries_a, (name,))
            cache_b = self._cache_from_components(morgan_b, summaries_b, (name,))
            cache_context_a = self.cache_input_projection(cache_a).mean(dim=1)
            cache_context_b = self.cache_input_projection(cache_b).mean(dim=1)
            pair = self._pair_features(
                summaries_a[name] + cache_context_a[:, None, :],
                summaries_b[name] + cache_context_b[:, None, :],
            ).mean(dim=1)
            correction = DualHeadOutput(
                self.modality_organ_heads[name](pair),
                self.modality_specific_heads[name](pair),
            )
            corrections[name] = correction
            gates = torch.sigmoid(self.modality_gate_logits[name])
            gate_values[name] = gates
            organ_logits = organ_logits + gates[0] * correction.organ_logits
            specific_logits = specific_logits + gates[1] * correction.specific_logits
        self._last_diagnostics = {
            "gates": {name: value.detach().clone() for name, value in gate_values.items()},
            "coverage": {
                name: torch.cat([coverage_a[name], coverage_b[name]]).detach().clone()
                for name in coverage_a
            },
        }
        output = DualHeadOutput(organ_logits, specific_logits)
        if not torch.isfinite(output.organ_logits).all() or not torch.isfinite(output.specific_logits).all():
            raise RuntimeError("multimodal teacher emitted non-finite logits")
        return output, corrections, baseline

    def diagnostics(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            key: {name: value.detach().clone() for name, value in values.items()}
            for key, values in self._last_diagnostics.items()
        }

    def validation_variants(
        self, drug_a: Mapping[str, Any], drug_b: Mapping[str, Any]
    ) -> dict[str, DualHeadOutput]:
        combined, corrections, baseline = self._forward_components(drug_a, drug_b)
        variants: dict[str, DualHeadOutput] = {"baseline": baseline}
        for name, correction in corrections.items():
            gates = torch.sigmoid(self.modality_gate_logits[name])
            variants[name] = DualHeadOutput(
                baseline.organ_logits + gates[0] * correction.organ_logits,
                baseline.specific_logits + gates[1] * correction.specific_logits,
            )
        if self.enabled_modalities:
            variants["combined"] = combined
        return variants

    def forward(self, drug_a: Mapping[str, Any], drug_b: Mapping[str, Any]) -> DualHeadOutput:
        if self.training_stage == "baseline":
            return self.morgan_baseline_logits(drug_a, drug_b)
        output, _, _ = self._forward_components(drug_a, drug_b)
        return output

    def set_training_stage(self, stage: str) -> None:
        if stage not in {"baseline", "fused"}:
            raise ValueError("training stage must be 'baseline' or 'fused'")
        self.training_stage = stage

    def baseline_parameters(self) -> list[nn.Parameter]:
        modules = (
            self.morgan_projection,
            self.morgan_missing_token,
            self.morgan_baseline_organ_head,
            self.morgan_baseline_specific_head,
        )
        parameters: list[nn.Parameter] = []
        for module in modules:
            if isinstance(module, nn.Parameter):
                parameters.append(module)
            else:
                parameters.extend(module.parameters())
        return parameters

    def auxiliary_parameters(self) -> list[nn.Parameter]:
        if not self.enabled_modalities:
            return []
        baseline_ids = {id(parameter) for parameter in self.baseline_parameters()}
        return [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in baseline_ids and parameter.requires_grad
        ]

    def freeze_baseline(self) -> None:
        for parameter in self.baseline_parameters():
            parameter.requires_grad_(False)

    def unfreeze_baseline(self) -> None:
        for parameter in self.baseline_parameters():
            parameter.requires_grad_(True)


class DistilledPairStudent(nn.Module):
    """Small pair model that consumes only cached latent drug tokens."""

    def __init__(
        self,
        token_dim: int,
        token_count: int,
        hidden_dim: int,
        num_organ: int,
        num_specific: int,
        *,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        token_dim = _positive_int(token_dim, "token_dim")
        token_count = _positive_int(token_count, "token_count")
        hidden_dim = _positive_int(hidden_dim, "hidden_dim")
        num_organ = _positive_int(num_organ, "num_organ")
        num_specific = _positive_int(num_specific, "num_specific")
        num_heads = _positive_int(num_heads, "num_heads")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        dropout = _validate_float(dropout, "dropout")
        if dropout >= 1.0:
            raise ValueError("dropout must be less than 1")
        self.token_dim = token_dim
        self.token_count = token_count
        self.hidden_dim = hidden_dim
        self.num_organ = num_organ
        self.num_specific = num_specific
        self.projection = nn.Linear(token_dim, hidden_dim)
        self.missing_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.label_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.organ_queries = nn.Parameter(torch.randn(num_organ, hidden_dim) * 0.02)
        self.specific_queries = nn.Parameter(torch.randn(num_specific, hidden_dim) * 0.02)
        self.organ_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        self.specific_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def _tokens(self, tokens: torch.Tensor, availability: torch.Tensor | None) -> torch.Tensor:
        tokens = _check_tensor(tokens, "cached_tokens", 3)
        if tokens.shape[1:] != (self.token_count, self.token_dim):
            raise ValueError(
                f"cached_tokens must have shape [batch, {self.token_count}, {self.token_dim}]"
            )
        valid = (
            torch.ones(tokens.shape[0], self.token_count, dtype=torch.bool, device=tokens.device)
            if availability is None
            else availability
        )
        if not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool:
            raise ValueError("token availability must be a boolean tensor")
        if valid.shape != (tokens.shape[0], self.token_count):
            raise ValueError(
                f"token availability must have shape [batch, {self.token_count}]"
            )
        projected = self.projection(tokens)
        return torch.where(valid[:, :, None], projected, self.missing_token)

    def _ordered_pair(self, drug_a: torch.Tensor, drug_b: torch.Tensor) -> DualHeadOutput:
        attended_a, _ = self.cross_attention(drug_a, drug_b, drug_b, need_weights=False)
        attended_b, _ = self.cross_attention(drug_b, drug_a, drug_a, need_weights=False)
        pair_tokens = torch.cat([drug_a + attended_a, drug_b + attended_b], dim=1)
        organ_queries = self.organ_queries.unsqueeze(0).expand(drug_a.shape[0], -1, -1)
        specific_queries = self.specific_queries.unsqueeze(0).expand(drug_a.shape[0], -1, -1)
        organ_context, _ = self.label_attention(organ_queries, pair_tokens, pair_tokens, need_weights=False)
        specific_context, _ = self.label_attention(
            specific_queries, pair_tokens, pair_tokens, need_weights=False
        )
        return DualHeadOutput(
            self.organ_head(organ_context).squeeze(-1),
            self.specific_head(specific_context).squeeze(-1),
        )

    def forward(
        self,
        drug_a_tokens: torch.Tensor,
        drug_b_tokens: torch.Tensor,
        availability_a: torch.Tensor | None = None,
        availability_b: torch.Tensor | None = None,
        *,
        mask_a: torch.Tensor | None = None,
        mask_b: torch.Tensor | None = None,
    ) -> DualHeadOutput:
        if availability_a is not None and mask_a is not None:
            raise ValueError("provide only one of availability_a and mask_a")
        if availability_b is not None and mask_b is not None:
            raise ValueError("provide only one of availability_b and mask_b")
        availability_a = availability_a if availability_a is not None else mask_a
        availability_b = availability_b if availability_b is not None else mask_b
        tokens_a = self._tokens(drug_a_tokens, availability_a)
        tokens_b = self._tokens(drug_b_tokens, availability_b)
        if tokens_a.shape[0] != tokens_b.shape[0]:
            raise ValueError("drug A and drug B must have the same batch size")
        first = self._ordered_pair(tokens_a, tokens_b)
        second = self._ordered_pair(tokens_b, tokens_a)
        return DualHeadOutput(
            (first.organ_logits + second.organ_logits) / 2.0,
            (first.specific_logits + second.specific_logits) / 2.0,
        )


class DistillationLoss(nn.Module):
    """Supervised BCE plus temperature-scaled teacher soft targets."""

    def __init__(
        self,
        *,
        temperature: float = 2.0,
        supervised_weight: float = 1.0,
        distillation_weight: float = 1.0,
        hierarchy_weight: float = 0.0,
        hierarchy_loss: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.temperature = _validate_float(temperature, "temperature", minimum=1e-6)
        self.supervised_weight = _validate_float(supervised_weight, "supervised_weight")
        self.distillation_weight = _validate_float(distillation_weight, "distillation_weight")
        self.hierarchy_weight = _validate_float(hierarchy_weight, "hierarchy_weight")
        if self.hierarchy_weight and hierarchy_loss is None:
            raise ValueError("hierarchy_loss is required when hierarchy_weight is non-zero")
        self.hierarchy_loss = hierarchy_loss

    def forward(
        self,
        student_output: DualHeadOutput,
        teacher_output: DualHeadOutput,
        *,
        organ_targets: torch.Tensor,
        specific_targets: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(student_output, DualHeadOutput) or not isinstance(teacher_output, DualHeadOutput):
            raise TypeError("student_output and teacher_output must be DualHeadOutput values")
        for name, value, expected in (
            ("organ_targets", organ_targets, student_output.organ_logits.shape),
            ("specific_targets", specific_targets, student_output.specific_logits.shape),
        ):
            if not isinstance(value, torch.Tensor) or value.shape != expected:
                raise ValueError(f"{name} must have shape {tuple(expected)}")
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite floating-point values")
            if (value < 0).any() or (value > 1).any():
                raise ValueError(f"{name} must contain values in [0, 1]")
        if teacher_output.organ_logits.shape != student_output.organ_logits.shape:
            raise ValueError("teacher and student organ logits must have equal shapes")
        if teacher_output.specific_logits.shape != student_output.specific_logits.shape:
            raise ValueError("teacher and student specific logits must have equal shapes")
        teacher_organ = teacher_output.organ_logits.detach()
        teacher_specific = teacher_output.specific_logits.detach()
        temperature = self.temperature
        supervised = F.binary_cross_entropy_with_logits(
            student_output.organ_logits, organ_targets
        ) + F.binary_cross_entropy_with_logits(student_output.specific_logits, specific_targets)
        if self.distillation_weight == 0.0:
            if self.hierarchy_weight:
                return supervised + self.hierarchy_weight * self.hierarchy_loss(
                    student_output.specific_logits, student_output.organ_logits
                )
            return self.supervised_weight * supervised
        soft = F.binary_cross_entropy_with_logits(
            student_output.organ_logits / temperature,
            torch.sigmoid(teacher_organ / temperature),
        ) + F.binary_cross_entropy_with_logits(
            student_output.specific_logits / temperature,
            torch.sigmoid(teacher_specific / temperature),
        )
        soft = soft * (temperature**2)
        total = self.supervised_weight * supervised + self.distillation_weight * soft
        if self.hierarchy_weight:
            total = total + self.hierarchy_weight * self.hierarchy_loss(
                student_output.specific_logits, student_output.organ_logits
            )
        return total


def distillation_loss(
    student_output: DualHeadOutput,
    teacher_output: DualHeadOutput,
    *,
    organ_targets: torch.Tensor,
    specific_targets: torch.Tensor,
    temperature: float = 2.0,
    supervised_weight: float = 1.0,
    distillation_weight: float = 1.0,
    hierarchy_weight: float = 0.0,
    hierarchy_loss: nn.Module | None = None,
) -> torch.Tensor:
    """Functional convenience wrapper around :class:`DistillationLoss`."""

    return DistillationLoss(
        temperature=temperature,
        supervised_weight=supervised_weight,
        distillation_weight=distillation_weight,
        hierarchy_weight=hierarchy_weight,
        hierarchy_loss=hierarchy_loss,
    )(
        student_output,
        teacher_output,
        organ_targets=organ_targets,
        specific_targets=specific_targets,
    )

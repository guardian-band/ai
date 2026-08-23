"""Stable tensor-only ONNX wrappers for the frozen warm Teacher.

The training model accepts nested dictionaries and performs Python-side input
validation.  Those behaviours are useful during training but are not a stable
ONNX boundary.  These wrappers expose only the tensors used by the selected
Morgan+MPNN release and reproduce the exact frozen forward computation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.multimodal_teacher_student import MultiModalTeacher


FUSED_INPUT_NAMES = (
    "morgan_a",
    "mpnn_tokens_a",
    "mpnn_padding_mask_a",
    "morgan_b",
    "mpnn_tokens_b",
    "mpnn_padding_mask_b",
)
BASELINE_INPUT_NAMES = ("morgan_a", "morgan_b")
OUTPUT_NAMES = ("organ_logits", "specific_logits")


class _TeacherWrapper(nn.Module):
    def __init__(self, teacher: MultiModalTeacher) -> None:
        super().__init__()
        if tuple(teacher.enabled_modalities) != ("mpnn",):
            raise ValueError("ONNX release requires enabled_modalities=['mpnn']")
        self.teacher = teacher

    @staticmethod
    def _pair_features(drug_a: torch.Tensor, drug_b: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [drug_a + drug_b, torch.abs(drug_a - drug_b), drug_a * drug_b], dim=-1
        )

    def _morgan(self, values: torch.Tensor) -> torch.Tensor:
        return self.teacher.morgan_projection(values)

    def _baseline(
        self, embedding_a: torch.Tensor, embedding_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self._pair_features(embedding_a, embedding_b)
        return (
            self.teacher.morgan_baseline_organ_head(pair),
            self.teacher.morgan_baseline_specific_head(pair),
        )


class FrozenTeacherBaselineOnnx(_TeacherWrapper):
    """Morgan fallback graph used when either MPNN feature is unavailable."""

    def forward(
        self, morgan_a: torch.Tensor, morgan_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._baseline(self._morgan(morgan_a), self._morgan(morgan_b))


class FrozenTeacherFusedOnnx(_TeacherWrapper):
    """Selected Morgan+MPNN Teacher graph for supported warm drug pairs."""

    def _mpnn_summary(
        self, values: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        projected = self.teacher.modality_projections["mpnn"](values)
        valid = ~padding_mask
        missing = valid.sum(dim=1) == 0

        # Tensor-only equivalent of the training model's all-padding fallback.
        valid = torch.cat((valid[:, :1] | missing[:, None], valid[:, 1:]), dim=1)
        first = torch.where(
            missing[:, None],
            self.teacher.modality_missing_tokens["mpnn"].view(1, -1),
            projected[:, 0],
        )
        projected = torch.cat((first[:, None, :], projected[:, 1:, :]), dim=1)
        projected = projected.masked_fill((~valid).unsqueeze(-1), 0.0)
        queries = self.teacher.modality_queries["mpnn"].unsqueeze(0).expand(
            values.shape[0], -1, -1
        )
        summary, _ = self.teacher.modality_resamplers["mpnn"](
            queries,
            projected,
            projected,
            key_padding_mask=~valid,
            need_weights=False,
        )
        return self.teacher.modality_summary_norms["mpnn"](summary)

    def _cache_context(
        self, morgan: torch.Tensor, summary: torch.Tensor
    ) -> torch.Tensor:
        components = torch.cat((morgan.unsqueeze(1), summary), dim=1)
        queries = self.teacher.cache_queries.unsqueeze(0).expand(
            components.shape[0], -1, -1
        )
        latent, _ = self.teacher.cache_resampler(
            queries, components, components, need_weights=False
        )
        tokens = self.teacher.cache_output_projection(latent)
        return self.teacher.cache_input_projection(tokens).mean(dim=1)

    def forward(
        self,
        morgan_a: torch.Tensor,
        mpnn_tokens_a: torch.Tensor,
        mpnn_padding_mask_a: torch.Tensor,
        morgan_b: torch.Tensor,
        mpnn_tokens_b: torch.Tensor,
        mpnn_padding_mask_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        morgan_embedding_a = self._morgan(morgan_a)
        morgan_embedding_b = self._morgan(morgan_b)
        baseline_organ, baseline_specific = self._baseline(
            morgan_embedding_a, morgan_embedding_b
        )

        summary_a = self._mpnn_summary(mpnn_tokens_a, mpnn_padding_mask_a)
        summary_b = self._mpnn_summary(mpnn_tokens_b, mpnn_padding_mask_b)
        context_a = self._cache_context(morgan_embedding_a, summary_a)
        context_b = self._cache_context(morgan_embedding_b, summary_b)
        pair = self._pair_features(
            summary_a + context_a[:, None, :],
            summary_b + context_b[:, None, :],
        ).mean(dim=1)
        correction_organ = self.teacher.modality_organ_heads["mpnn"](pair)
        correction_specific = self.teacher.modality_specific_heads["mpnn"](pair)
        gates = torch.sigmoid(self.teacher.modality_gate_logits["mpnn"])
        return (
            baseline_organ + gates[0] * correction_organ,
            baseline_specific + gates[1] * correction_specific,
        )

"""Hierarchy consistency utilities for separate organ and specific heads."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class HierarchyMapping:
    """Resolved specific-CUI to organ-index mapping.

    ``organ_order`` is deterministic and is part of the mapping contract so
    callers can construct organ targets without relying on dictionary order.
    """

    specific_to_organ: dict[str, int]
    organ_order: tuple[str, ...]
    source: str = "versioned_json"

    @property
    def mapping(self) -> dict[str, int]:
        """Compatibility alias for callers that expect ``mapping``."""

        return dict(self.specific_to_organ)


def load_hierarchy_mapping(
    path: str | Path,
    *,
    selected_specific_cuis: Iterable[str] | None = None,
) -> HierarchyMapping:
    """Load and validate a versioned CUI hierarchy mapping.

    The supported JSON shape is ``{"schema_version": 1, "mappings": [...]}``,
    where each record has ``specific_cui`` and either ``organ_cui`` or
    ``organ_index``.  For index records, ``organ_order`` (or ``organs``) is
    required.  CUI records use the explicitly supplied order when present and
    otherwise a lexical order, never an arbitrary modulo assignment.
    """

    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read hierarchy mapping {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Hierarchy mapping must be a JSON object")
    version = payload.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ValueError("Hierarchy mapping schema_version must be 1")
    records = payload.get("mappings", payload.get("records"))
    if not isinstance(records, list) or not records:
        raise ValueError("Hierarchy mapping must contain a non-empty mappings list")

    raw_order = payload.get("organ_order", payload.get("organs"))
    if raw_order is not None:
        if not isinstance(raw_order, list) or not raw_order or any(
            not isinstance(item, str) or not item.strip() for item in raw_order
        ):
            raise ValueError("organ_order must be a non-empty list of names")
        organ_order = tuple(item.strip() for item in raw_order)
        if len(set(organ_order)) != len(organ_order):
            raise ValueError("organ_order entries must be unique")
    else:
        organ_order = ()

    specific_to_organ_name: dict[str, str] = {}
    specific_to_organ_index: dict[str, int] = {}
    saw_cui = False
    saw_index = False
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Hierarchy mapping records must be objects")
        specific = record.get("specific_cui")
        if not isinstance(specific, str) or not specific.strip():
            raise ValueError("Each hierarchy mapping requires specific_cui")
        specific = specific.strip()
        if specific in specific_to_organ_name or specific in specific_to_organ_index:
            raise ValueError(f"Duplicate hierarchy mapping for {specific}")
        has_cui = "organ_cui" in record
        has_index = "organ_index" in record
        if has_cui == has_index:
            raise ValueError("Each record must contain exactly one of organ_cui or organ_index")
        if has_cui:
            organ_cui = record.get("organ_cui")
            if not isinstance(organ_cui, str) or not organ_cui.strip():
                raise ValueError(f"Invalid organ_cui for {specific}")
            specific_to_organ_name[specific] = organ_cui.strip()
            saw_cui = True
        else:
            organ_index = record.get("organ_index")
            if isinstance(organ_index, bool) or not isinstance(organ_index, int) or organ_index < 0:
                raise ValueError(f"Invalid organ_index for {specific}")
            specific_to_organ_index[specific] = organ_index
            saw_index = True

    if saw_cui and saw_index:
        raise ValueError("Hierarchy mapping cannot mix organ_cui and organ_index records")
    if saw_cui:
        if not organ_order:
            organ_order = tuple(sorted(set(specific_to_organ_name.values())))
        unknown = set(specific_to_organ_name.values()) - set(organ_order)
        if unknown:
            raise ValueError(f"organ_order is missing mapped organs: {sorted(unknown)}")
        resolved = {
            specific: organ_order.index(organ)
            for specific, organ in specific_to_organ_name.items()
        }
    else:
        if not organ_order:
            raise ValueError("organ_order is required when records use organ_index")
        if any(index >= len(organ_order) for index in specific_to_organ_index.values()):
            raise ValueError("organ_index is out of bounds for organ_order")
        resolved = dict(specific_to_organ_index)

    if selected_specific_cuis is not None:
        selected = [str(cui) for cui in selected_specific_cuis]
        missing = sorted(set(selected) - set(resolved))
        if missing:
            raise ValueError(f"Hierarchy mapping is not complete for selected labels: {missing}")

    return HierarchyMapping(resolved, tuple(organ_order), source="versioned_json")


class HierarchyLoss(nn.Module):
    """Penalize specific probabilities above their parent organ probability.

    The two heads are intentionally separate: ``forward`` requires
    ``specific_logits [B, N_specific]`` and ``organ_logits [B, N_organ]``.
    ``mapping`` maps specific-column indices to organ-column indices.
    """

    def __init__(
        self,
        mapping: Mapping[int, int],
        *,
        num_specific: int | None = None,
        num_organ: int | None = None,
        required_specific_indices: Iterable[int] | None = None,
    ):
        super().__init__()
        normalized: dict[int, int] = {}
        for specific_index, organ_index in mapping.items():
            if (
                isinstance(specific_index, bool)
                or not isinstance(specific_index, int)
                or specific_index < 0
                or isinstance(organ_index, bool)
                or not isinstance(organ_index, int)
                or organ_index < 0
            ):
                raise ValueError("Hierarchy mapping indices must be non-negative integers")
            normalized[specific_index] = organ_index
        required = set(range(num_specific)) if num_specific is not None else set(
            required_specific_indices or ()
        )
        if num_specific is not None and (isinstance(num_specific, bool) or num_specific < 0):
            raise ValueError("num_specific must be a non-negative integer")
        if num_organ is not None and (isinstance(num_organ, bool) or num_organ < 0):
            raise ValueError("num_organ must be a non-negative integer")
        missing = required - set(normalized)
        if missing:
            raise ValueError(f"Hierarchy mapping must provide a complete mapping; missing {sorted(missing)}")
        if num_specific is not None and any(index >= num_specific for index in normalized):
            raise ValueError("specific hierarchy index is out of bounds")
        if num_organ is not None and any(index >= num_organ for index in normalized.values()):
            raise ValueError("organ hierarchy index is out of bounds")
        self.mapping = dict(normalized)
        self.specific_indices = tuple(normalized)
        self.parent_indices = tuple(normalized[index] for index in self.specific_indices)
        self.num_specific = num_specific
        self.num_organ = num_organ

    def forward(
        self,
        specific_logits: torch.Tensor,
        organ_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if organ_logits is None:
            raise TypeError("HierarchyLoss.forward requires separate specific_logits and organ_logits")
        if not isinstance(specific_logits, torch.Tensor) or not isinstance(organ_logits, torch.Tensor):
            raise TypeError("specific_logits and organ_logits must be torch tensors")
        if specific_logits.ndim != 2 or organ_logits.ndim != 2:
            raise ValueError("specific_logits and organ_logits must be 2D tensors")
        if specific_logits.shape[0] != organ_logits.shape[0]:
            raise ValueError("specific and organ heads must have the same batch size")
        if self.num_specific is not None and specific_logits.shape[1] != self.num_specific:
            raise ValueError("specific_logits has the wrong number of columns")
        if self.num_organ is not None and organ_logits.shape[1] != self.num_organ:
            raise ValueError("organ_logits has the wrong number of columns")
        if self.specific_indices and max(self.specific_indices) >= specific_logits.shape[1]:
            raise ValueError("specific hierarchy index is out of bounds")
        if self.parent_indices and max(self.parent_indices) >= organ_logits.shape[1]:
            raise ValueError("organ hierarchy index is out of bounds")
        if not self.specific_indices:
            return specific_logits.sum() * 0.0
        specific_probs = torch.sigmoid(specific_logits[:, self.specific_indices])
        parent_probs = torch.sigmoid(organ_logits[:, self.parent_indices])
        return F.relu(specific_probs - parent_probs).mean()


# Existing keyword-based MedDRA mapping is intentionally labelled as a fallback
# and should not be used as authoritative hierarchy evidence.
def keyword_fallback_mapping(side_effects_raw_path):
    from src.features.meddra_hierarchy import build_meddra_hierarchical_mapping

    cui_to_soc, soc_categories = build_meddra_hierarchical_mapping(side_effects_raw_path)
    organ_order = tuple(soc_categories)
    return HierarchyMapping(
        {
            str(cui): organ_order.index(str(soc))
            for cui, soc in cui_to_soc.items()
            if str(soc) in organ_order
        },
        organ_order,
        source="keyword_fallback",
    )

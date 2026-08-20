"""Strict typed PrimeKG preprocessing for leakage-safe graph encoders."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd
import torch

from src.models.primekg_hgt import split_relation_edges


LEAKAGE_RELATIONS = frozenset(
    {
        "drug_drug",
        "drug_effect",
        "contraindication",
        "indication",
        "off_label_use",
        "synergistic_interaction",
    }
)
REQUIRED_COLUMNS = frozenset({"x_id", "x_type", "relation", "y_id", "y_type"})


@dataclass(frozen=True)
class PreparedTypedPrimeKG:
    node_ids: Mapping[str, tuple[str, ...]]
    edge_index: Mapping[tuple[str, str, str], torch.Tensor]
    rejected_leakage_edges: int
    duplicate_edges: int

    @property
    def metadata(self) -> tuple[tuple[str, ...], tuple[tuple[str, str, str], ...]]:
        return tuple(sorted(self.node_ids)), tuple(sorted(self.edge_index))


def prepare_typed_primekg(frame: pd.DataFrame) -> PreparedTypedPrimeKG:
    """Preserve PrimeKG types, remove target leakage, and add reverse relations."""

    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"PrimeKG is missing required columns: {', '.join(missing)}")
    columns = ["x_id", "x_type", "relation", "y_id", "y_type"]
    values = frame.loc[:, columns].copy()
    if values.isna().any().any():
        raise ValueError("PrimeKG required fields must not contain null values")
    for column in ("x_id", "y_id"):
        values[column] = values[column].astype(str).str.strip()
        if values[column].eq("").any() or values[column].str.lower().eq("nan").any():
            raise ValueError("PrimeKG node IDs must be non-empty strings")
    for column in ("x_type", "relation", "y_type"):
        values[column] = (
            values[column]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"[^a-z0-9]+", "_", regex=True)
            .str.strip("_")
        )
        if values[column].eq("").any():
            raise ValueError(f"PrimeKG {column} values cannot normalize to empty strings")
    leakage = values["relation"].isin(LEAKAGE_RELATIONS)
    rejected = int(leakage.sum())
    safe = values.loc[~leakage]
    before_deduplication = len(safe)
    safe = safe.drop_duplicates(ignore_index=True)
    duplicate_edges = before_deduplication - len(safe)
    if safe.empty:
        raise ValueError("PrimeKG has no safe typed edges after leakage filtering")
    node_types = sorted(set(safe["x_type"]) | set(safe["y_type"]))
    node_ids: dict[str, tuple[str, ...]] = {}
    indexes: dict[str, pd.Index] = {}
    for node_type in node_types:
        ids = np.concatenate(
            [
                safe.loc[safe["x_type"] == node_type, "x_id"].to_numpy(),
                safe.loc[safe["y_type"] == node_type, "y_id"].to_numpy(),
            ]
        )
        ordered = tuple(sorted(pd.unique(ids).tolist()))
        node_ids[node_type] = ordered
        indexes[node_type] = pd.Index(ordered)
    edge_index: dict[tuple[str, str, str], torch.Tensor] = {}
    grouped = safe.groupby(["x_type", "relation", "y_type"], sort=True, observed=True)
    for (source_type, relation, destination_type), group in grouped:
        source = indexes[source_type].get_indexer(group["x_id"])
        destination = indexes[destination_type].get_indexer(group["y_id"])
        if (source < 0).any() or (destination < 0).any():
            raise RuntimeError("PrimeKG node index construction failed")
        edges = torch.from_numpy(np.stack([source, destination])).to(dtype=torch.long)
        edge_type = (str(source_type), str(relation), str(destination_type))
        reverse_type = (str(destination_type), f"rev_{relation}", str(source_type))
        edge_index[edge_type] = edges.contiguous()
        edge_index[reverse_type] = edges.flip(0).contiguous()
    return PreparedTypedPrimeKG(
        MappingProxyType(node_ids),
        MappingProxyType(edge_index),
        rejected,
        duplicate_edges,
    )


def typed_edge_index_sha256(
    edge_index: Mapping[tuple[str, str, str], torch.Tensor],
) -> str:
    """Hash a typed graph canonically, independent of mapping or edge order."""

    digest = hashlib.sha256()
    for edge_type in sorted(edge_index):
        if (
            not isinstance(edge_type, tuple)
            or len(edge_type) != 3
            or not all(isinstance(item, str) and item for item in edge_type)
        ):
            raise ValueError("edge types must be non-empty three-part string tuples")
        edges = edge_index[edge_type]
        if edges.dtype != torch.long or edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("edge indices must be int64 with shape [2, E]")
        values = edges.detach().cpu().numpy().astype("<i8", copy=False)
        if values.shape[1]:
            order = np.lexsort((values[1], values[0]))
            values = values[:, order]
        values = np.ascontiguousarray(values)
        digest.update("\x1f".join(edge_type).encode("utf-8"))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def validate_typed_edge_index_sha256(
    edge_index: Mapping[tuple[str, str, str], torch.Tensor], expected_sha256: str
) -> None:
    """Fail closed when a declared typed message graph hash differs."""

    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("declared graph hash must be a canonical SHA-256 hash")
    actual = typed_edge_index_sha256(edge_index)
    if actual != expected_sha256:
        raise ValueError(
            f"message graph hash mismatch: declared {expected_sha256}, actual {actual}"
        )


def build_leakage_safe_edge_partitions(
    edge_index: Mapping[tuple[str, str, str], torch.Tensor],
    *,
    seed: int,
    train_ratio: float = 0.1,
    validation_ratio: float = 0.1,
) -> tuple[
    dict[tuple[str, str, str], torch.Tensor],
    dict[tuple[str, str, str], torch.Tensor],
    dict[tuple[str, str, str], torch.Tensor],
]:
    """Split forward edges once and derive reverse message edges from that split."""

    message: dict[tuple[str, str, str], torch.Tensor] = {}
    train: dict[tuple[str, str, str], torch.Tensor] = {}
    validation: dict[tuple[str, str, str], torch.Tensor] = {}
    forward_types = sorted(item for item in edge_index if not item[1].startswith("rev_"))
    if not forward_types:
        raise ValueError("typed graph has no forward relations to supervise")
    for edge_type in forward_types:
        source_type, relation, destination_type = edge_type
        reverse_type = (destination_type, f"rev_{relation}", source_type)
        if reverse_type not in edge_index:
            raise ValueError(f"typed graph is missing reverse relation for {edge_type}")
        forward_edges = edge_index[edge_type]
        expected_reverse = forward_edges.flip(0)
        actual_reverse = edge_index[reverse_type]
        expected_pairs = set(map(tuple, expected_reverse.t().tolist()))
        actual_pairs = set(map(tuple, actual_reverse.t().tolist()))
        if expected_pairs != actual_pairs:
            raise ValueError(f"reverse relation does not exactly match {edge_type}")
        relation_digest = hashlib.sha256("\x1f".join(edge_type).encode()).digest()
        relation_seed = seed ^ int.from_bytes(relation_digest[:8], "big")
        if forward_edges.shape[1] < 3:
            message_edges = forward_edges.clone()
            train_edges = forward_edges.new_empty((2, 0))
            validation_edges = forward_edges.new_empty((2, 0))
        else:
            message_edges, train_edges, validation_edges = split_relation_edges(
                forward_edges,
                seed=relation_seed,
                train_ratio=train_ratio,
                validation_ratio=validation_ratio,
            )
        message[edge_type] = message_edges
        message[reverse_type] = message_edges.flip(0).contiguous()
        if train_edges.shape[1]:
            train[edge_type] = train_edges
        if validation_edges.shape[1]:
            validation[edge_type] = validation_edges
    return message, train, validation

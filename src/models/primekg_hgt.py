"""Typed heterogeneous graph transformer used by the PrimeKG producer."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from src.features.token_feature_artifact import TokenFeatureArtifact


EdgeType = tuple[str, str, str]


def split_relation_edges(
    edge_index: torch.Tensor,
    *,
    seed: int,
    train_ratio: float = 0.1,
    validation_ratio: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split known edges into message, train-supervision, and validation sets."""

    if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must be int64 with shape [2, E]")
    count = edge_index.shape[1]
    if count < 3:
        raise ValueError("at least three edges are required for a leakage-safe split")
    if not 0.0 < train_ratio < 1.0 or not 0.0 < validation_ratio < 1.0:
        raise ValueError("split ratios must be in (0, 1)")
    if train_ratio + validation_ratio >= 1.0:
        raise ValueError("train and validation ratios must leave message-passing edges")
    pairs = edge_index.t()
    if torch.unique(pairs, dim=0).shape[0] != count:
        raise ValueError("edge_index contains duplicate edges")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(count, generator=generator)
    train_count = max(1, int(round(count * train_ratio)))
    validation_count = max(1, int(round(count * validation_ratio)))
    if train_count + validation_count >= count:
        validation_count = 1
        train_count = 1
    train_indices = order[:train_count]
    validation_indices = order[train_count : train_count + validation_count]
    message_indices = order[train_count + validation_count :]
    return (
        edge_index[:, message_indices].contiguous(),
        edge_index[:, train_indices].contiguous(),
        edge_index[:, validation_indices].contiguous(),
    )


def sample_bipartite_negatives(
    all_positive_edges: torch.Tensor,
    *,
    num_source_nodes: int,
    num_destination_nodes: int,
    count: int,
    seed: int,
) -> torch.Tensor:
    """Sample unique negatives after exact rejection against every known positive."""

    if (
        all_positive_edges.dtype != torch.long
        or all_positive_edges.ndim != 2
        or all_positive_edges.shape[0] != 2
    ):
        raise ValueError("all_positive_edges must be int64 with shape [2, E]")
    if min(num_source_nodes, num_destination_nodes, count) <= 0:
        raise ValueError("node counts and requested negative count must be positive")
    positives = {
        int(source) * num_destination_nodes + int(destination)
        for source, destination in all_positive_edges.t().tolist()
    }
    capacity = num_source_nodes * num_destination_nodes - len(positives)
    if count > capacity:
        raise ValueError("requested negatives exceed the available bipartite complement")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    selected: set[int] = set()
    while len(selected) < count:
        remaining = count - len(selected)
        source = torch.randint(
            num_source_nodes, (max(remaining * 2, 32),), generator=generator
        )
        destination = torch.randint(
            num_destination_nodes, (source.numel(),), generator=generator
        )
        for encoded in (source * num_destination_nodes + destination).tolist():
            if encoded not in positives:
                selected.add(int(encoded))
                if len(selected) == count:
                    break
    ordered = sorted(selected)
    return torch.tensor(
        [
            [value // num_destination_nodes for value in ordered],
            [value % num_destination_nodes for value in ordered],
        ],
        dtype=torch.long,
    )


def _relation_key(edge_type: EdgeType) -> str:
    text = "\x1f".join(edge_type)
    return "r_" + hashlib.sha256(text.encode()).hexdigest()[:16]


def _segment_softmax(
    scores: torch.Tensor, destination: torch.Tensor, node_count: int
) -> torch.Tensor:
    heads = scores.shape[1]
    expanded_index = destination[:, None].expand(-1, heads)
    maxima = scores.new_full((node_count, heads), -torch.inf)
    maxima.scatter_reduce_(0, expanded_index, scores, reduce="amax", include_self=True)
    exponentials = torch.exp(scores - maxima[destination])
    denominator = scores.new_zeros((node_count, heads))
    denominator.index_add_(0, destination, exponentials)
    return exponentials / denominator[destination].clamp_min(torch.finfo(scores.dtype).tiny)


class _HGTLayer(nn.Module):
    def __init__(
        self,
        node_types: Sequence[str],
        edge_types: Sequence[EdgeType],
        hidden_dim: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.edge_types = tuple(edge_types)
        self.query = nn.ModuleDict({node_type: nn.Linear(hidden_dim, hidden_dim) for node_type in node_types})
        self.key = nn.ModuleDict({node_type: nn.Linear(hidden_dim, hidden_dim) for node_type in node_types})
        self.value = nn.ModuleDict({node_type: nn.Linear(hidden_dim, hidden_dim) for node_type in node_types})
        self.output = nn.ModuleDict({node_type: nn.Linear(hidden_dim, hidden_dim) for node_type in node_types})
        self.norm = nn.ModuleDict({node_type: nn.LayerNorm(hidden_dim) for node_type in node_types})
        self.relation_key = nn.ParameterDict()
        self.relation_value = nn.ParameterDict()
        self.relation_prior = nn.ParameterDict()
        for edge_type in edge_types:
            key = _relation_key(edge_type)
            self.relation_key[key] = nn.Parameter(
                torch.eye(self.head_dim).repeat(heads, 1, 1)
            )
            self.relation_value[key] = nn.Parameter(
                torch.eye(self.head_dim).repeat(heads, 1, 1)
            )
            self.relation_prior[key] = nn.Parameter(torch.ones(heads))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: Mapping[str, torch.Tensor],
        edge_index_dict: Mapping[EdgeType, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        queries = {
            node_type: self.query[node_type](values).view(-1, self.heads, self.head_dim)
            for node_type, values in x_dict.items()
        }
        keys = {
            node_type: self.key[node_type](values).view(-1, self.heads, self.head_dim)
            for node_type, values in x_dict.items()
        }
        values = {
            node_type: self.value[node_type](node_values).view(
                -1, self.heads, self.head_dim
            )
            for node_type, node_values in x_dict.items()
        }
        incoming: dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = defaultdict(list)
        for edge_type in self.edge_types:
            if edge_type not in edge_index_dict:
                continue
            source_type, _, destination_type = edge_type
            if source_type not in x_dict or destination_type not in x_dict:
                continue
            edge_index = edge_index_dict[edge_type]
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError(f"edge index for {edge_type} must have shape [2, E]")
            if edge_index.shape[1] == 0:
                continue
            source, destination = edge_index
            relation = _relation_key(edge_type)
            relation_keys = torch.einsum(
                "ehd,hdf->ehf", keys[source_type][source], self.relation_key[relation]
            )
            relation_values = torch.einsum(
                "ehd,hdf->ehf", values[source_type][source], self.relation_value[relation]
            )
            scores = (
                (queries[destination_type][destination] * relation_keys).sum(dim=-1)
                / math.sqrt(self.head_dim)
            ) * self.relation_prior[relation]
            incoming[destination_type].append((scores, relation_values, destination))

        output: dict[str, torch.Tensor] = {}
        for node_type, original in x_dict.items():
            if node_type not in incoming:
                output[node_type] = self.norm[node_type](original)
                continue
            scores = torch.cat([item[0] for item in incoming[node_type]], dim=0)
            messages = torch.cat([item[1] for item in incoming[node_type]], dim=0)
            destination = torch.cat([item[2] for item in incoming[node_type]], dim=0)
            weights = _segment_softmax(scores, destination, original.shape[0])
            aggregate = original.new_zeros(
                (original.shape[0], self.heads, self.head_dim)
            )
            aggregate.index_add_(0, destination, weights.unsqueeze(-1) * messages)
            updated = self.output[node_type](aggregate.reshape(-1, self.hidden_dim))
            output[node_type] = self.norm[node_type](original + self.dropout(updated))
        return output


class PrimeKGHGT(nn.Module):
    """HGT-style typed attention encoder with layerwise drug token export."""

    def __init__(
        self,
        metadata: tuple[Sequence[str], Sequence[EdgeType]],
        *,
        input_dims: Mapping[str, int],
        hidden_dim: int = 256,
        layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        node_types = tuple(metadata[0])
        edge_types = tuple(tuple(item) for item in metadata[1])
        if not node_types or not edge_types or "drug" not in node_types:
            raise ValueError("metadata must include drug nodes and at least one typed edge")
        if hidden_dim <= 0 or layers <= 0 or heads <= 0 or hidden_dim % heads:
            raise ValueError("hidden_dim/layers/heads must be positive and heads must divide hidden_dim")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        missing = sorted(set(node_types) - set(input_dims))
        if missing:
            raise ValueError(f"missing input dimensions for node types: {', '.join(missing)}")
        self.node_types = node_types
        self.edge_types = edge_types
        self.input_dims = {node_type: int(input_dims[node_type]) for node_type in node_types}
        self.hidden_dim = hidden_dim
        self.num_layers = layers
        self.heads = heads
        self.dropout_probability = float(dropout)
        self.input_projection = nn.ModuleDict(
            {node_type: nn.Linear(int(input_dims[node_type]), hidden_dim) for node_type in node_types}
        )
        self.layers = nn.ModuleList(
            [_HGTLayer(node_types, edge_types, hidden_dim, heads, dropout) for _ in range(layers)]
        )

    def forward(
        self,
        x_dict: Mapping[str, torch.Tensor],
        edge_index_dict: Mapping[EdgeType, torch.Tensor],
        *,
        return_drug_tokens: bool = False,
    ):
        if not x_dict or not set(x_dict).issubset(self.node_types):
            raise ValueError("x_dict contains no nodes or node types outside HGT metadata")
        hidden = {
            node_type: self.input_projection[node_type](x_dict[node_type])
            for node_type in self.node_types
        }
        drug_layers = [hidden["drug"]] if "drug" in hidden else []
        for layer in self.layers:
            hidden = layer(hidden, edge_index_dict)
            if "drug" in hidden:
                drug_layers.append(hidden["drug"])
        if return_drug_tokens:
            if not drug_layers:
                raise ValueError("return_drug_tokens requires drug nodes in x_dict")
            return hidden, torch.stack(drug_layers, dim=1)
        return hidden


class TypedLinkPredictor(nn.Module):
    """Relation-specific bilinear decoder for HGT pretraining."""

    def __init__(self, edge_types: Sequence[EdgeType], hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.hidden_dim = int(hidden_dim)
        self.relations = nn.ParameterDict(
            {
                _relation_key(edge_type): nn.Parameter(torch.ones(hidden_dim))
                for edge_type in edge_types
            }
        )
        self.relation_bias = nn.ParameterDict(
            {
                _relation_key(edge_type): nn.Parameter(torch.zeros(()))
                for edge_type in edge_types
            }
        )

    def forward(
        self,
        x_dict: Mapping[str, torch.Tensor],
        edge_type: EdgeType,
        edge_label_index: torch.Tensor,
    ) -> torch.Tensor:
        source_type, _, destination_type = edge_type
        source, destination = edge_label_index
        relation = self.relations[_relation_key(edge_type)]
        bias = self.relation_bias[_relation_key(edge_type)]
        score = (
            x_dict[source_type][source] * relation * x_dict[destination_type][destination]
        ).sum(dim=-1) / math.sqrt(self.hidden_dim)
        score = score + bias
        if not torch.isfinite(score).all():
            raise FloatingPointError("TypedLinkPredictor produced non-finite logits")
        return score


def save_hgt_checkpoint(
    model: PrimeKGHGT,
    path: str | Path,
    *,
    source_sha256: str,
    training_config: Mapping[str, Any],
    predictor: TypedLinkPredictor | None = None,
) -> str:
    """Atomically save an HGT checkpoint loadable with ``weights_only=True``."""

    if not isinstance(source_sha256, str) or len(source_sha256) != 64:
        raise ValueError("source_sha256 must be a canonical SHA-256 hash")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "model_config": {
            "node_types": list(model.node_types),
            "edge_types": [list(item) for item in model.edge_types],
            "input_dims": model.input_dims,
            "hidden_dim": model.hidden_dim,
            "layers": model.num_layers,
            "heads": model.heads,
            "dropout": model.dropout_probability,
        },
        "source_sha256": source_sha256,
        "training_config": dict(training_config),
        "state_dict": {
            key: value.detach().to(device="cpu") for key, value in model.state_dict().items()
        },
    }
    if predictor is not None:
        payload["predictor_state_dict"] = {
            key: value.detach().to(device="cpu")
            for key, value in predictor.state_dict().items()
        }
    file_descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(file_descriptor)
    try:
        torch.save(payload, temporary)
        data = Path(temporary).read_bytes()
        checksum = hashlib.sha256(data).hexdigest()
        os.replace(temporary, path)
        path.with_name(f"{path.name}.sha256").write_text(
            checksum + "\n", encoding="ascii"
        )
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return checksum


def load_hgt_checkpoint(
    path: str | Path, *, expected_sha256: str | None = None
) -> tuple[PrimeKGHGT, dict[str, Any]]:
    path = Path(path)
    try:
        data = path.read_bytes()
        sidecar = path.with_name(f"{path.name}.sha256").read_text(encoding="ascii").strip()
    except OSError as exc:
        raise ValueError(f"unable to read HGT checkpoint {path}") from exc
    actual = hashlib.sha256(data).hexdigest()
    if actual != sidecar or (expected_sha256 is not None and actual != expected_sha256):
        raise ValueError("HGT checkpoint checksum verification failed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported HGT checkpoint schema")
    config = payload.get("model_config")
    state_dict = payload.get("state_dict")
    if not isinstance(config, dict) or not isinstance(state_dict, dict):
        raise ValueError("HGT checkpoint is missing model config or weights")
    node_types = tuple(config["node_types"])
    edge_types = tuple(tuple(item) for item in config["edge_types"])
    model = PrimeKGHGT(
        (node_types, edge_types),
        input_dims=config["input_dims"],
        hidden_dim=config["hidden_dim"],
        layers=config["layers"],
        heads=config["heads"],
        dropout=config["dropout"],
    )
    model.load_state_dict(state_dict, strict=True)
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"state_dict", "predictor_state_dict"}
    }
    return model, metadata


def export_hgt_token_artifact(
    model: PrimeKGHGT,
    *,
    drug_ids: Sequence[str],
    x_dict: Mapping[str, torch.Tensor],
    edge_index_dict: Mapping[EdgeType, torch.Tensor],
    output_path: str | Path,
    source_sha256: str,
    checkpoint_sha256: str,
    device: str | torch.device = "cpu",
) -> TokenFeatureArtifact:
    """Export layerwise HGT drug states from an in-memory inference graph."""

    ids = list(drug_ids)
    if len(ids) != x_dict.get("drug", torch.empty(0)).shape[0]:
        raise ValueError("drug_ids must align exactly with drug node features")
    model = model.to(device).eval()
    device_x = {key: value.to(device) for key, value in x_dict.items()}
    device_edges = {key: value.to(device) for key, value in edge_index_dict.items()}
    with torch.inference_mode():
        _, tokens = model(device_x, device_edges, return_drug_tokens=True)
    token_values = tokens.to(dtype=torch.float32, device="cpu").numpy()
    return TokenFeatureArtifact.write(
        output_path,
        ids,
        np.ascontiguousarray(token_values),
        np.zeros(token_values.shape[:2], dtype=bool),
        np.ones(len(ids), dtype=bool),
        producer="primekg_hgt",
        producer_config={
            "hidden_dim": model.hidden_dim,
            "layers": model.num_layers,
            "heads": model.heads,
            "token_semantics": "input_projection_plus_each_hgt_layer",
        },
        source_sha256=source_sha256,
        checkpoint_sha256=checkpoint_sha256,
    )

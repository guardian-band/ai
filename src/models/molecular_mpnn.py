"""Deterministic RDKit graph featurization and bond-aware molecular MPNN."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import os
import math
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from src.features.token_feature_artifact import TokenFeatureArtifact


ATOM_FEATURE_DIM = 15
BOND_FEATURE_DIM = 6
MAX_ATOMIC_NUMBER = 118


@dataclass(frozen=True)
class MolecularGraph:
    atom_features: torch.Tensor
    atom_numbers: torch.Tensor
    edge_index: torch.Tensor
    bond_features: torch.Tensor


@dataclass(frozen=True)
class MolecularGraphBatch:
    atom_features: torch.Tensor
    atom_numbers: torch.Tensor
    edge_index: torch.Tensor
    bond_features: torch.Tensor
    graph_index: torch.Tensor
    graph_ptr: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.graph_ptr.numel() - 1)


def _one_hot(index: int, size: int) -> list[float]:
    values = [0.0] * size
    if 0 <= index < size:
        values[index] = 1.0
    return values


def _atom_features(atom) -> list[float]:
    hybridization_values = [
        str(atom.GetHybridization()) == name
        for name in ("SP", "SP2", "SP3", "SP3D", "SP3D2")
    ]
    return [
        atom.GetAtomicNum() / MAX_ATOMIC_NUMBER,
        min(atom.GetTotalDegree(), 6) / 6.0,
        max(-4, min(4, atom.GetFormalCharge())) / 4.0,
        min(atom.GetTotalNumHs(), 4) / 4.0,
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        *[float(value) for value in hybridization_values],
        *_one_hot(min(atom.GetChiralTag().real, 3), 4),
    ]


def _bond_features(bond) -> list[float]:
    bond_type = str(bond.GetBondType())
    return [
        float(bond_type == "SINGLE"),
        float(bond_type == "DOUBLE"),
        float(bond_type == "TRIPLE"),
        float(bond_type == "AROMATIC"),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


def featurize_smiles(smiles: str) -> MolecularGraph:
    """Canonicalize SMILES and emit deterministic directed molecular edges."""

    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("invalid SMILES: expected a non-empty string")
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    canonical = Chem.MolToSmiles(molecule, canonical=True)
    molecule = Chem.MolFromSmiles(canonical)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError(f"invalid SMILES: {smiles}")
    atom_features = torch.tensor(
        [_atom_features(atom) for atom in molecule.GetAtoms()], dtype=torch.float32
    )
    atom_numbers = torch.tensor(
        [atom.GetAtomicNum() for atom in molecule.GetAtoms()], dtype=torch.long
    )
    edges: list[tuple[int, int]] = []
    features: list[list[float]] = []
    for bond in molecule.GetBonds():
        first = bond.GetBeginAtomIdx()
        second = bond.GetEndAtomIdx()
        bond_vector = _bond_features(bond)
        edges.extend(((first, second), (second, first)))
        features.extend((bond_vector, bond_vector))
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        bond_features = torch.tensor(features, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        bond_features = torch.empty((0, BOND_FEATURE_DIM), dtype=torch.float32)
    return MolecularGraph(atom_features, atom_numbers, edge_index, bond_features)


def collate_molecular_graphs(graphs: Sequence[MolecularGraph]) -> MolecularGraphBatch:
    if not graphs:
        raise ValueError("at least one molecular graph is required")
    atom_features: list[torch.Tensor] = []
    atom_numbers: list[torch.Tensor] = []
    edges: list[torch.Tensor] = []
    bond_features: list[torch.Tensor] = []
    graph_indices: list[torch.Tensor] = []
    ptr = [0]
    offset = 0
    for graph_index, graph in enumerate(graphs):
        count = int(graph.atom_features.shape[0])
        if graph.atom_features.shape != (count, ATOM_FEATURE_DIM):
            raise ValueError("atom_features has an invalid shape")
        if graph.atom_numbers.shape != (count,):
            raise ValueError("atom_numbers has an invalid shape")
        atom_features.append(graph.atom_features)
        atom_numbers.append(graph.atom_numbers)
        edges.append(graph.edge_index + offset)
        bond_features.append(graph.bond_features)
        graph_indices.append(torch.full((count,), graph_index, dtype=torch.long))
        offset += count
        ptr.append(offset)
    return MolecularGraphBatch(
        atom_features=torch.cat(atom_features, dim=0),
        atom_numbers=torch.cat(atom_numbers, dim=0),
        edge_index=torch.cat(edges, dim=1),
        bond_features=torch.cat(bond_features, dim=0),
        graph_index=torch.cat(graph_indices, dim=0),
        graph_ptr=torch.tensor(ptr, dtype=torch.long),
    )


class MolecularMPNN(nn.Module):
    """Bond-conditioned message passing with fixed atom-token export."""

    def __init__(self, hidden_dim: int = 256, layers: int = 4, token_count: int = 32) -> None:
        super().__init__()
        for value, name in (
            (hidden_dim, "hidden_dim"),
            (layers, "layers"),
            (token_count, "token_count"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.token_count = token_count
        self.atom_encoder = nn.Linear(ATOM_FEATURE_DIM, hidden_dim)
        self.bond_encoder = nn.Linear(BOND_FEATURE_DIM, hidden_dim)
        self.message_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(layers)
            ]
        )
        self.updates = nn.ModuleList([nn.GRUCell(hidden_dim, hidden_dim) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.atom_classifier = nn.Linear(hidden_dim, MAX_ATOMIC_NUMBER)

    def encode_nodes(self, batch: MolecularGraphBatch) -> torch.Tensor:
        device = self.atom_encoder.weight.device
        atom_features = batch.atom_features.to(device)
        edge_index = batch.edge_index.to(device)
        bond_features = batch.bond_features.to(device)
        hidden = self.atom_encoder(atom_features)
        if edge_index.shape[1] == 0:
            return hidden
        source, destination = edge_index
        bond_hidden = self.bond_encoder(bond_features)
        for message_layer, update, norm in zip(
            self.message_layers, self.updates, self.norms, strict=True
        ):
            messages = message_layer(torch.cat([hidden[source], bond_hidden], dim=-1))
            aggregate = hidden.new_zeros(hidden.shape)
            aggregate.index_add_(0, destination, messages)
            degree = hidden.new_zeros((hidden.shape[0], 1))
            degree.index_add_(
                0,
                destination,
                torch.ones((destination.numel(), 1), device=device, dtype=hidden.dtype),
            )
            aggregate = aggregate / degree.clamp_min(1.0)
            hidden = norm(update(aggregate, hidden))
        return hidden

    def encode_tokens(self, batch: MolecularGraphBatch) -> tuple[torch.Tensor, torch.Tensor]:
        atom_counts = [
            int(batch.graph_ptr[index + 1] - batch.graph_ptr[index])
            for index in range(batch.batch_size)
        ]
        if atom_counts and max(atom_counts) > self.token_count:
            raise ValueError(
                f"token_count={self.token_count} cannot encode {max(atom_counts)} atoms; "
                f"required minimum token_count={max(atom_counts)}"
            )
        hidden = self.encode_nodes(batch)
        tokens = hidden.new_zeros((batch.batch_size, self.token_count, self.hidden_dim))
        padding_mask = torch.ones(
            (batch.batch_size, self.token_count), dtype=torch.bool, device=hidden.device
        )
        for graph_index in range(batch.batch_size):
            start = int(batch.graph_ptr[graph_index])
            stop = int(batch.graph_ptr[graph_index + 1])
            count = stop - start
            tokens[graph_index, :count] = hidden[start : start + count]
            padding_mask[graph_index, :count] = False
        return tokens, padding_mask


def masked_atom_loss(
    model: MolecularMPNN,
    batch: MolecularGraphBatch,
    *,
    mask_probability: float = 0.15,
    generator_seed: int,
) -> torch.Tensor:
    """Self-supervised atom-identity loss used to pretrain the MPNN producer."""

    if not math.isfinite(mask_probability) or not 0.0 < mask_probability <= 1.0:
        raise ValueError("mask_probability must be in (0, 1]")
    generator = torch.Generator(device="cpu").manual_seed(generator_seed)
    selected = torch.rand(batch.atom_numbers.shape, generator=generator) < mask_probability
    if not bool(selected.any()):
        selected[torch.randint(len(selected), (1,), generator=generator)] = True
    masked_features = batch.atom_features.clone()
    masked_features[selected] = 0.0
    masked_batch = replace(batch, atom_features=masked_features)
    hidden = model.encode_nodes(masked_batch)
    targets = batch.atom_numbers.to(hidden.device)[selected.to(hidden.device)] - 1
    logits = model.atom_classifier(hidden[selected.to(hidden.device)])
    return F.cross_entropy(logits, targets)


def save_mpnn_checkpoint(
    model: MolecularMPNN,
    path: str | Path,
    *,
    source_sha256: str,
    training_config: Mapping[str, Any],
) -> str:
    """Atomically save a weights-only-compatible MPNN checkpoint and checksum."""

    if not isinstance(source_sha256, str) or len(source_sha256) != 64:
        raise ValueError("source_sha256 must be a canonical SHA-256 hash")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "model_config": {
            "hidden_dim": model.hidden_dim,
            "layers": model.layers,
            "token_count": model.token_count,
        },
        "source_sha256": source_sha256,
        "training_config": dict(training_config),
        "state_dict": {
            key: value.detach().to(device="cpu") for key, value in model.state_dict().items()
        },
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
        checksum_path = path.with_name(f"{path.name}.sha256")
        checksum_path.write_text(checksum + "\n", encoding="ascii")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return checksum


def load_mpnn_checkpoint(
    path: str | Path, *, expected_sha256: str | None = None
) -> tuple[MolecularMPNN, dict[str, Any]]:
    path = Path(path)
    try:
        data = path.read_bytes()
        sidecar = path.with_name(f"{path.name}.sha256").read_text(encoding="ascii").strip()
    except OSError as exc:
        raise ValueError(f"unable to read MPNN checkpoint {path}") from exc
    actual = hashlib.sha256(data).hexdigest()
    if actual != sidecar or (expected_sha256 is not None and actual != expected_sha256):
        raise ValueError("MPNN checkpoint checksum verification failed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported MPNN checkpoint schema")
    config = payload.get("model_config")
    state_dict = payload.get("state_dict")
    if not isinstance(config, dict) or not isinstance(state_dict, dict):
        raise ValueError("MPNN checkpoint is missing model config or weights")
    model = MolecularMPNN(**config)
    model.load_state_dict(state_dict, strict=True)
    metadata = {key: value for key, value in payload.items() if key != "state_dict"}
    return model, metadata


def export_mpnn_token_artifact(
    model: MolecularMPNN,
    *,
    drug_ids: Sequence[str],
    smiles: Sequence[str],
    output_path: str | Path,
    source_sha256: str,
    checkpoint_sha256: str,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
    upstream_validation_loss: float | None = None,
) -> TokenFeatureArtifact:
    """Export deterministic fixed atom tokens from a trained MPNN."""

    ids = list(drug_ids)
    values = list(smiles)
    if not ids or len(ids) != len(values):
        raise ValueError("drug_ids and smiles must be non-empty aligned sequences")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    graphs: list[MolecularGraph] = []
    valid_indices: list[int] = []
    available = np.zeros(len(ids), dtype=bool)
    for index, value in enumerate(values):
        try:
            graph = featurize_smiles(value)
        except ValueError:
            continue
        graphs.append(graph)
        valid_indices.append(index)
        available[index] = True
    if not graphs:
        raise ValueError("at least one valid SMILES is required for MPNN export")
    atom_counts = [int(graph.atom_features.shape[0]) for graph in graphs]
    insufficient = [
        (ids[index], count)
        for index, count in zip(valid_indices, atom_counts, strict=True)
        if count > model.token_count
    ]
    if insufficient:
        minimum = max(count for _, count in insufficient)
        affected = ", ".join(f"{drug_id} ({count})" for drug_id, count in insufficient)
        raise ValueError(
            f"MPNN token_count={model.token_count} is insufficient; required minimum "
            f"token_count={minimum}; affected drug IDs: {affected}"
        )
    if upstream_validation_loss is not None and not math.isfinite(float(upstream_validation_loss)):
        raise ValueError("upstream_validation_loss must be finite when provided")
    distribution = {
        "counts": {
            str(count): atom_counts.count(count) for count in sorted(set(atom_counts))
        },
        "max": max(atom_counts),
        "mean": float(np.mean(atom_counts)),
        "min": min(atom_counts),
    }
    model = model.to(device).eval()
    all_tokens = np.zeros(
        (len(ids), model.token_count, model.hidden_dim), dtype=np.float32
    )
    all_padding_mask = np.ones((len(ids), model.token_count), dtype=bool)
    with torch.inference_mode():
        for start in range(0, len(graphs), batch_size):
            batch = collate_molecular_graphs(graphs[start : start + batch_size])
            tokens, padding_mask = model.encode_tokens(batch)
            batch_indices = valid_indices[start : start + batch.batch_size]
            all_tokens[batch_indices] = tokens.to(
                dtype=torch.float32, device="cpu"
            ).numpy()
            all_padding_mask[batch_indices] = padding_mask.to(device="cpu").numpy()
    return TokenFeatureArtifact.write(
        output_path,
        ids,
        np.ascontiguousarray(all_tokens, dtype=np.float32),
        np.ascontiguousarray(all_padding_mask, dtype=bool),
        available,
        producer="molecular_mpnn",
        producer_config={
            "hidden_dim": model.hidden_dim,
            "layers": model.layers,
            "token_count": model.token_count,
            "pretraining_objective": "masked_atom_identity",
            "atom_token_count_distribution": distribution,
            "truncation_count": 0,
            "source_sha256": source_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "upstream_validation_loss": (
                None
                if upstream_validation_loss is None
                else float(upstream_validation_loss)
            ),
        },
        source_sha256=source_sha256,
        checkpoint_sha256=checkpoint_sha256,
    )

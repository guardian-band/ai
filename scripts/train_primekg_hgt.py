"""Train a typed, leakage-safe PrimeKG HGT and export benchmark drug tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data.primekg_typed_graph import (
    PreparedTypedPrimeKG,
    build_leakage_safe_edge_partitions,
    prepare_typed_primekg,
    typed_edge_index_sha256,
    validate_typed_edge_index_sha256,
)
from src.features.token_feature_artifact import TokenFeatureArtifact
from src.models.primekg_hgt import (
    PrimeKGHGT,
    TypedLinkPredictor,
    sample_bipartite_negatives,
    save_hgt_checkpoint,
)
from src.training.engine import verify_manifest


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def _resolve(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def _hgt_graph_provenance(
    message_graph: Mapping[tuple[str, str, str], torch.Tensor],
    train_edges: Mapping[tuple[str, str, str], torch.Tensor],
    validation_edges: Mapping[tuple[str, str, str], torch.Tensor],
    *,
    excluded_drug_ids: set[str],
    required_drug_ids: tuple[str, ...],
    drug_ids: tuple[str, ...],
) -> dict[str, object]:
    """Build auditable provenance for the exact HGT message-passing graph."""

    relation_edge_counts: dict[str, dict[str, int]] = {}
    relation_types = set(message_graph) | set(train_edges) | set(validation_edges)
    for edge_type in sorted(relation_types):
        relation_edge_counts[_relation_name(edge_type)] = {
            "message": int(message_graph.get(edge_type, torch.empty(2, 0)).shape[1]),
            "train": int(train_edges.get(edge_type, torch.empty(2, 0)).shape[1]),
            "validation": int(
                validation_edges.get(edge_type, torch.empty(2, 0)).shape[1]
            ),
        }
    drug_index = {drug_id: index for index, drug_id in enumerate(drug_ids)}
    connected_drug_indices: set[int] = set()
    for (source_type, _, destination_type), edges in message_graph.items():
        if source_type == "drug":
            connected_drug_indices.update(int(item) for item in edges[0].tolist())
        if destination_type == "drug":
            connected_drug_indices.update(int(item) for item in edges[1].tolist())
    required = tuple(sorted(set(required_drug_ids)))
    covered_count = sum(
        1 for drug_id in required if drug_index.get(drug_id) in connected_drug_indices
    )
    required_count = len(required)
    excluded_digest = hashlib.sha256(
        json.dumps(sorted(excluded_drug_ids), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "message_graph_sha256": typed_edge_index_sha256(message_graph),
        "excluded_drug_ids_sha256": excluded_digest,
        "excluded_drug_count": len(excluded_drug_ids),
        "excluded_cold_drug_count": len(excluded_drug_ids),
        "relation_edge_counts": relation_edge_counts,
        "kg_neighbor_coverage_count": covered_count,
        "kg_neighbor_coverage_total": required_count,
        "kg_neighbor_coverage_percentage": (
            100.0 * covered_count / required_count if required_count else 0.0
        ),
        "cold_protocol": "train_message_graph_only",
    }


def _validate_hgt_message_graph_artifact(
    artifact: TokenFeatureArtifact,
    message_graph: Mapping[tuple[str, str, str], torch.Tensor],
) -> None:
    """Validate that an HGT artifact declares the graph used for token export."""

    producer_config = artifact.metadata.get("producer_config")
    if not isinstance(producer_config, Mapping):
        raise ValueError("HGT artifact producer_config is missing")
    if producer_config.get("cold_protocol") != "train_message_graph_only":
        raise ValueError("HGT artifact has an invalid cold graph protocol")
    expected = producer_config.get("message_graph_sha256")
    if not isinstance(expected, str):
        raise ValueError("HGT artifact is missing its declared message graph hash")
    validate_typed_edge_index_sha256(message_graph, expected)


def _training_metrics_paths(checkpoint: Path) -> tuple[Path, Path]:
    """Return metrics and checksum paths uniquely scoped to one checkpoint."""

    metrics = checkpoint.with_name(f"{checkpoint.name}.training_metrics.json")
    return metrics, metrics.with_name(f"{metrics.name}.sha256")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_training_metrics(path: Path, payload: Mapping[str, object]) -> str:
    """Atomically persist metrics and its SHA-256 sidecar."""

    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    checksum = hashlib.sha256(data).hexdigest()
    _atomic_write_bytes(path, data)
    sidecar = path.with_name(f"{path.name}.sha256")
    _atomic_write_bytes(sidecar, f"{checksum}\n".encode("ascii"))
    return checksum


def _excluded_cold_drugs(manifest_path: Path, manifest: Mapping[str, object]) -> set[str]:
    pairs = pd.read_parquet(_resolve(manifest_path, str(manifest["pairs_path"])))
    train = pairs[pairs["split"] == "train"]
    train_drugs = set(train["drug_a"].astype(str)) | set(train["drug_b"].astype(str))
    all_drugs = set(pairs["drug_a"].astype(str)) | set(pairs["drug_b"].astype(str))
    return set() if manifest["scenario"] == "warm_pair" else all_drugs - train_drugs


def _filter_incident_drugs(
    prepared: PreparedTypedPrimeKG, excluded: set[str]
) -> dict[tuple[str, str, str], torch.Tensor]:
    if not excluded:
        return {key: value.clone() for key, value in prepared.edge_index.items()}
    drug_ids = prepared.node_ids.get("drug", ())
    excluded_indices = {
        index for index, drug_id in enumerate(drug_ids) if drug_id in excluded
    }
    excluded_tensor = torch.tensor(sorted(excluded_indices), dtype=torch.long)
    output: dict[tuple[str, str, str], torch.Tensor] = {}
    for edge_type, edge_index in prepared.edge_index.items():
        source_type, _, destination_type = edge_type
        keep = torch.ones(edge_index.shape[1], dtype=torch.bool)
        if source_type == "drug" and excluded_tensor.numel():
            keep &= ~torch.isin(edge_index[0], excluded_tensor)
        if destination_type == "drug" and excluded_tensor.numel():
            keep &= ~torch.isin(edge_index[1], excluded_tensor)
        output[edge_type] = edge_index[:, keep].contiguous()
    return output


def _structural_features(
    node_ids: Mapping[str, tuple[str, ...]],
    edge_index: Mapping[tuple[str, str, str], torch.Tensor],
) -> dict[str, torch.Tensor]:
    degrees = {
        node_type: torch.zeros((len(ids), 3), dtype=torch.float32)
        for node_type, ids in node_ids.items()
    }
    for (source_type, _, destination_type), edges in edge_index.items():
        if edges.shape[1] == 0:
            continue
        degrees[source_type][:, 1].index_add_(
            0, edges[0], torch.ones(edges.shape[1], dtype=torch.float32)
        )
        degrees[destination_type][:, 0].index_add_(
            0, edges[1], torch.ones(edges.shape[1], dtype=torch.float32)
        )
    return {
        node_type: torch.cat(
            [torch.log1p(values), torch.ones((values.shape[0], 1))], dim=1
        )
        for node_type, values in degrees.items()
    }


def _node_features(
    prepared: PreparedTypedPrimeKG,
    edge_index: Mapping[tuple[str, str, str], torch.Tensor],
    morgan_path: Path,
    drug_id_column: str,
    morgan_column: str,
) -> dict[str, torch.Tensor]:
    structural = _structural_features(prepared.node_ids, edge_index)
    frame = pd.read_parquet(morgan_path)
    missing = sorted({drug_id_column, morgan_column} - set(frame.columns))
    if missing:
        raise ValueError(f"Morgan artifact is missing columns: {', '.join(missing)}")
    mapping = {
        str(row[drug_id_column]): np.asarray(row[morgan_column], dtype=np.float32)
        for row in frame.to_dict("records")
    }
    if not mapping:
        raise ValueError("Morgan artifact is empty")
    dimension = len(next(iter(mapping.values())))
    drug_values = np.zeros((len(prepared.node_ids["drug"]), dimension), dtype=np.float32)
    for index, drug_id in enumerate(prepared.node_ids["drug"]):
        value = mapping.get(drug_id)
        if value is not None:
            if value.shape != (dimension,) or not np.isfinite(value).all():
                raise ValueError(f"invalid Morgan row for {drug_id}")
            drug_values[index] = value
    structural["drug"] = torch.cat(
        [torch.from_numpy(drug_values), structural["drug"]], dim=1
    )
    return structural


def _heterodata(x_dict, edge_index_dict):
    try:
        from torch_geometric.data import HeteroData
    except ImportError as exc:
        raise RuntimeError("train_primekg_hgt requires torch-geometric") from exc
    data = HeteroData()
    for node_type, values in x_dict.items():
        data[node_type].x = values
    for edge_type, values in edge_index_dict.items():
        data[edge_type].edge_index = values
    return data


def _labels(
    positives: torch.Tensor,
    all_positives: torch.Tensor,
    source_count: int,
    destination_count: int,
    seed: int,
    *,
    negatives: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if negatives is None:
        negatives = sample_bipartite_negatives(
            all_positives,
            num_source_nodes=source_count,
            num_destination_nodes=destination_count,
            count=positives.shape[1],
            seed=seed,
        )
    if negatives.dtype != torch.long or negatives.ndim != 2 or negatives.shape[0] != 2:
        raise ValueError("negative edges must be int64 with shape [2, E]")
    if negatives.shape[1] != positives.shape[1]:
        raise ValueError("negative and positive edge counts must match")
    return (
        torch.cat([positives, negatives], dim=1),
        torch.cat([torch.ones(positives.shape[1]), torch.zeros(negatives.shape[1])]),
    )


def _bounded_supervision(
    positives: torch.Tensor,
    max_edges: int | None,
    *,
    seed: int,
) -> torch.Tensor:
    """Deterministically cap one relation without dropping the relation itself."""

    if positives.ndim != 2 or positives.shape[0] != 2:
        raise ValueError("positive edge index must have shape [2, E]")
    if max_edges is None:
        return positives
    if isinstance(max_edges, bool) or not isinstance(max_edges, int) or max_edges <= 0:
        raise ValueError("max supervision edges must be a positive integer")
    if positives.shape[1] <= max_edges:
        return positives
    generator = torch.Generator(device="cpu").manual_seed(seed)
    selection = torch.randperm(positives.shape[1], generator=generator)[:max_edges]
    return positives[:, selection].contiguous()


def _relation_name(edge_type: tuple[str, str, str]) -> str:
    return "\x1f".join(edge_type)


def _require_finite_tensor(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains non-finite values")


def _require_finite_mapping(values: Mapping[str, torch.Tensor], name: str) -> None:
    for key, value in values.items():
        _require_finite_tensor(value, f"{name}[{key}]")


def _binary_link_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float | int | None]:
    _require_finite_tensor(logits, "link logits")
    _require_finite_tensor(labels, "link labels")
    if logits.ndim != 1 or labels.ndim != 1 or logits.shape != labels.shape:
        raise ValueError("link logits and labels must be one-dimensional and aligned")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("link labels must contain only zero and one")
    if not logits.numel():
        raise ValueError("link metrics require at least one labelled edge")
    labels = labels.to(dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    _require_finite_tensor(loss, "link loss")
    labels_np = labels.detach().cpu().numpy().astype(np.int64)
    probabilities = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)
    positive = logits[labels == 1]
    negative = logits[labels == 0]

    def _stat(values: torch.Tensor, operation: str) -> float | None:
        if not values.numel():
            return None
        result = values.mean() if operation == "mean" else values.std(unbiased=False)
        _require_finite_tensor(result, f"{operation} logit statistic")
        return float(result.detach().cpu())

    ap = (
        float(average_precision_score(labels_np, probabilities))
        if int((labels == 1).sum()) > 0
        else None
    )
    auroc = (
        float(roc_auc_score(labels_np, probabilities))
        if np.unique(labels_np).size == 2
        else None
    )
    for name, value in (("AP", ap), ("AUROC", auroc)):
        if value is not None and not math.isfinite(value):
            raise FloatingPointError(f"{name} is non-finite")
    return {
        "bce": float(loss.detach().cpu()),
        "ap": ap,
        "auroc": auroc,
        "positive_logit_mean": _stat(positive, "mean"),
        "positive_logit_std": _stat(positive, "std"),
        "negative_logit_mean": _stat(negative, "mean"),
        "negative_logit_std": _stat(negative, "std"),
        "positive_count": int(positive.numel()),
        "negative_count": int(negative.numel()),
    }


def _link_prediction_metrics(
    relation_outputs: Mapping[
        tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]
    ],
) -> dict[str, object]:
    """Return auditable per-relation and global HGT link metrics."""

    if not relation_outputs:
        raise ValueError("link metrics require at least one relation")
    relation_metrics: dict[str, dict[str, float | int | None]] = {}
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for edge_type, (logits, labels) in sorted(relation_outputs.items()):
        metrics = _binary_link_metrics(logits, labels)
        relation_metrics[_relation_name(edge_type)] = metrics
        all_logits.append(logits.detach())
        all_labels.append(labels.detach())
    global_metrics = _binary_link_metrics(
        torch.cat(all_logits), torch.cat(all_labels)
    )
    aps = [
        float(metrics["ap"])
        for metrics in relation_metrics.values()
        if metrics["ap"] is not None
    ]
    if not aps:
        raise ValueError("no relation has defined average precision")
    macro_relation_ap = float(np.mean(aps))
    if not math.isfinite(macro_relation_ap):
        raise FloatingPointError("macro relation AP is non-finite")
    return {
        "relations": relation_metrics,
        "global": global_metrics,
        "macro_relation_ap": macro_relation_ap,
    }


def _fixed_validation_negatives(
    supervision: Mapping[tuple[str, str, str], torch.Tensor],
    all_edges: Mapping[tuple[str, str, str], torch.Tensor],
    node_ids: Mapping[str, tuple[str, ...]],
    *,
    seed: int,
    max_positive_edges: int | None = None,
) -> dict[tuple[str, str, str], torch.Tensor]:
    """Sample validation negatives once so every epoch uses identical controls."""

    negatives: dict[tuple[str, str, str], torch.Tensor] = {}
    for relation_number, (edge_type, positives) in enumerate(sorted(supervision.items())):
        positives = _bounded_supervision(
            positives,
            max_positive_edges,
            seed=seed + relation_number,
        )
        if positives.shape[1] == 0:
            continue
        source_type, _, destination_type = edge_type
        negatives[edge_type] = sample_bipartite_negatives(
            all_edges[edge_type],
            num_source_nodes=len(node_ids[source_type]),
            num_destination_nodes=len(node_ids[destination_type]),
            count=positives.shape[1],
            seed=seed + relation_number,
        )
    if not negatives:
        raise RuntimeError("validation contains no supervised relation edges")
    return negatives


def select_hgt_checkpoint(history: list[dict[str, object]]) -> dict[str, object]:
    """Select by macro relation AP, BCE, and epoch in that deterministic order."""

    candidates: list[dict[str, object]] = []
    for item in history:
        validation = item.get("validation")
        if not isinstance(validation, Mapping):
            raise ValueError("HGT history entry is missing validation metrics")
        macro_ap = validation.get("macro_relation_ap")
        bce = validation.get("bce")
        epoch = item.get("epoch")
        if not isinstance(macro_ap, (int, float)) or not math.isfinite(float(macro_ap)):
            continue
        if not isinstance(bce, (int, float)) or not math.isfinite(float(bce)):
            continue
        if not isinstance(epoch, int) or epoch <= 0:
            raise ValueError("HGT history epoch must be a positive integer")
        candidates.append(item)
    if not candidates:
        raise RuntimeError("HGT training produced no finite validation checkpoint")
    return min(
        candidates,
        key=lambda item: (
            -float(item["validation"]["macro_relation_ap"]),
            float(item["validation"]["bce"]),
            int(item["epoch"]),
        ),
    )


def _require_export_ready(validation: Mapping[str, object]) -> None:
    """Reject graph tokens when validation does not beat its negative baseline."""

    relation_metrics = validation.get("relations")
    global_metrics = validation.get("global")
    macro_ap = validation.get("macro_relation_ap")
    if not isinstance(relation_metrics, Mapping) or not isinstance(global_metrics, Mapping):
        raise ValueError("validation metrics are missing relation/global reports")
    if not isinstance(macro_ap, (int, float)) or not math.isfinite(float(macro_ap)):
        raise FloatingPointError("validation macro relation AP is non-finite")
    prevalences = []
    for metrics in relation_metrics.values():
        if not isinstance(metrics, Mapping):
            raise ValueError("invalid relation metrics")
        positive_count = metrics.get("positive_count")
        negative_count = metrics.get("negative_count")
        if not isinstance(positive_count, int) or not isinstance(negative_count, int):
            raise ValueError("relation metrics are missing class counts")
        total = positive_count + negative_count
        if positive_count and negative_count and total:
            prevalences.append(positive_count / total)
    if not prevalences or float(macro_ap) <= float(np.mean(prevalences)):
        raise RuntimeError(
            "HGT validation macro AP does not exceed the negative-sampling baseline"
        )
    bce = global_metrics.get("bce")
    if not isinstance(bce, (int, float)) or not math.isfinite(float(bce)):
        raise FloatingPointError("HGT validation BCE is non-finite")
    if float(bce) > 2.0:
        raise RuntimeError(
            "HGT validation BCE exceeds 2.0; investigate logit scaling before export"
        )
    positive_mean = global_metrics.get("positive_logit_mean")
    negative_mean = global_metrics.get("negative_logit_mean")
    if (
        not isinstance(positive_mean, (int, float))
        or not isinstance(negative_mean, (int, float))
        or not math.isfinite(float(positive_mean))
        or not math.isfinite(float(negative_mean))
        or float(positive_mean) <= float(negative_mean)
    ):
        raise RuntimeError("HGT validation positive logits do not rank above negatives")


def _relation_epoch(
    model,
    predictor,
    data,
    supervision,
    all_edges,
    node_ids,
    *,
    neighbors,
    batch_size,
    seed,
    device,
    optimizer=None,
    fixed_negatives: Mapping[tuple[str, str, str], torch.Tensor] | None = None,
    max_positive_edges: int | None = None,
    phase: str = "train",
) -> dict[str, object]:
    from torch_geometric.loader import LinkNeighborLoader

    outputs: dict[tuple[str, str, str], list[torch.Tensor]] = {}
    relations = sorted(supervision.items())
    for relation_number, (edge_type, positives) in enumerate(relations):
        positives = _bounded_supervision(
            positives,
            max_positive_edges,
            seed=seed + relation_number,
        )
        print(
            f"{phase} relation={relation_number + 1}/{len(relations)} "
            f"type={'/'.join(edge_type)} positives={positives.shape[1]}",
            flush=True,
        )
        source_type, _, destination_type = edge_type
        edge_label_index, edge_label = _labels(
            positives,
            all_edges[edge_type],
            len(node_ids[source_type]),
            len(node_ids[destination_type]),
            seed + relation_number,
            negatives=(fixed_negatives or {}).get(edge_type),
        )
        loader = LinkNeighborLoader(
            data,
            num_neighbors=neighbors,
            edge_label_index=(edge_type, edge_label_index),
            edge_label=edge_label,
            batch_size=batch_size,
            shuffle=optimizer is not None,
        )
        for batch in loader:
            batch = batch.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            output = model(batch.x_dict, batch.edge_index_dict)
            _require_finite_mapping(output, "HGT embeddings")
            logits = predictor(output, edge_type, batch[edge_type].edge_label_index)
            _require_finite_tensor(logits, "HGT link logits")
            loss = F.binary_cross_entropy_with_logits(logits, batch[edge_type].edge_label)
            _require_finite_tensor(loss, "HGT link loss")
            if optimizer is not None:
                loss.backward()
                for parameter in list(model.parameters()) + list(predictor.parameters()):
                    if parameter.grad is not None:
                        _require_finite_tensor(parameter.grad, "HGT gradient")
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(predictor.parameters()), 5.0
                )
                optimizer.step()
            outputs.setdefault(edge_type, [[], []])[0].append(logits.detach().cpu())
            outputs.setdefault(edge_type, [[], []])[1].append(
                batch[edge_type].edge_label.detach().cpu()
            )
    if not outputs:
        raise RuntimeError("HGT edge partitions contain no supervised examples")
    report = _link_prediction_metrics(
        {
            edge_type: (torch.cat(values[0]), torch.cat(values[1]))
            for edge_type, values in outputs.items()
        }
    )
    return report


def _relation_loss(
    model,
    predictor,
    data,
    supervision,
    all_edges,
    node_ids,
    *,
    neighbors,
    batch_size,
    seed,
    device,
    optimizer=None,
    fixed_negatives=None,
    max_positive_edges=None,
    phase="train",
) -> float:
    """Compatibility wrapper returning only global BCE."""

    report = _relation_epoch(
        model,
        predictor,
        data,
        supervision,
        all_edges,
        node_ids,
        neighbors=neighbors,
        batch_size=batch_size,
        seed=seed,
        device=device,
        optimizer=optimizer,
        fixed_negatives=fixed_negatives,
        max_positive_edges=max_positive_edges,
        phase=phase,
    )
    return float(report["global"]["bce"])


def _export_tokens(
    model,
    message_data,
    required_drugs,
    drug_ids,
    *,
    neighbors,
    batch_size,
    device,
):
    from torch_geometric.loader import NeighborLoader

    mapping = {drug_id: index for index, drug_id in enumerate(drug_ids)}
    present = [(drug_id, mapping[drug_id]) for drug_id in required_drugs if drug_id in mapping]
    token_count = model.num_layers + 1
    tokens = np.zeros((len(required_drugs), token_count, model.hidden_dim), dtype=np.float32)
    available = np.zeros(len(required_drugs), dtype=bool)
    required_index = {drug_id: index for index, drug_id in enumerate(required_drugs)}
    if present:
        seeds = torch.tensor([index for _, index in present], dtype=torch.long)
        loader = NeighborLoader(
            message_data,
            input_nodes=("drug", seeds),
            num_neighbors=neighbors,
            batch_size=batch_size,
            shuffle=False,
        )
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                batch = batch.to(device)
                _, batch_tokens = model(
                    batch.x_dict, batch.edge_index_dict, return_drug_tokens=True
                )
                _require_finite_tensor(batch_tokens, "HGT exported embeddings")
                seed_count = batch["drug"].batch_size
                global_ids = batch["drug"].n_id[:seed_count].cpu().tolist()
                for local_index, global_index in enumerate(global_ids):
                    drug_id = drug_ids[global_index]
                    output_index = required_index[drug_id]
                    tokens[output_index] = batch_tokens[local_index].cpu().numpy()
                    available[output_index] = True
    return tokens, available


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primekg", type=Path, required=True)
    parser.add_argument("--morgan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drug-id-column", default="drugbank_id")
    parser.add_argument("--morgan-column", default="morgan_fingerprint")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-neighbors", type=int, nargs="+", default=[15, 10, 5])
    parser.add_argument("--max-train-edges-per-relation", type=int)
    parser.add_argument("--max-validation-edges-per-relation", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    for value, name in (
        (args.max_train_edges_per_relation, "max-train-edges-per-relation"),
        (args.max_validation_edges_per_relation, "max-validation-edges-per-relation"),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    torch.manual_seed(args.seed)
    manifest_path = args.manifest.resolve()
    manifest = verify_manifest(str(manifest_path))
    pairs = pd.read_parquet(_resolve(manifest_path, str(manifest["pairs_path"])))
    required_drugs = sorted(
        set(pairs["drug_a"].astype(str)) | set(pairs["drug_b"].astype(str))
    )
    prepared = prepare_typed_primekg(_read(args.primekg))
    excluded = _excluded_cold_drugs(manifest_path, manifest)
    training_edges = _filter_incident_drugs(prepared, excluded)
    message, train_edges, validation_edges = build_leakage_safe_edge_partitions(
        training_edges, seed=args.seed
    )
    graph_provenance = _hgt_graph_provenance(
        message,
        train_edges,
        validation_edges,
        excluded_drug_ids=excluded,
        required_drug_ids=tuple(required_drugs),
        drug_ids=prepared.node_ids["drug"],
    )
    train_x = _node_features(
        prepared,
        message,
        args.morgan,
        args.drug_id_column,
        args.morgan_column,
    )
    train_data = _heterodata(train_x, message)
    input_dims = {key: value.shape[1] for key, value in train_x.items()}
    model = PrimeKGHGT(
        prepared.metadata,
        input_dims=input_dims,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(args.device)
    predictor = TypedLinkPredictor(tuple(train_edges), args.hidden_dim).to(args.device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(predictor.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    fixed_validation_negatives = _fixed_validation_negatives(
        validation_edges,
        training_edges,
        prepared.node_ids,
        seed=args.seed + 1_000_003,
        max_positive_edges=args.max_validation_edges_per_relation,
    )
    history: list[dict[str, object]] = []
    best_model = None
    best_predictor = None
    stale = 0
    for epoch in range(args.epochs):
        model.train()
        predictor.train()
        train_report = _relation_epoch(
            model,
            predictor,
            train_data,
            train_edges,
            training_edges,
            prepared.node_ids,
            neighbors=args.num_neighbors,
            batch_size=args.batch_size,
            seed=args.seed + epoch * 101,
            device=args.device,
            optimizer=optimizer,
            max_positive_edges=args.max_train_edges_per_relation,
            phase="train",
        )
        model.eval()
        predictor.eval()
        with torch.no_grad():
            validation_report = _relation_epoch(
                model,
                predictor,
                train_data,
                validation_edges,
                training_edges,
                prepared.node_ids,
                neighbors=args.num_neighbors,
                batch_size=args.batch_size,
                seed=args.seed + 1_000_003,
                device=args.device,
                fixed_negatives=fixed_validation_negatives,
                max_positive_edges=args.max_validation_edges_per_relation,
                phase="validation",
            )
        train_loss = float(train_report["global"]["bce"])
        validation_loss = float(validation_report["global"]["bce"])
        epoch_record = {
            "epoch": epoch + 1,
            "train": train_report,
            "validation": {**validation_report, "bce": validation_loss},
        }
        history.append(epoch_record)
        print(
            f"epoch={epoch + 1}/{args.epochs} train_loss={train_loss:.6f} "
            f"validation_loss={validation_loss:.6f} "
            f"validation_macro_relation_ap={validation_report['macro_relation_ap']:.6f}"
        )
        selected = select_hgt_checkpoint(history)
        if selected is epoch_record:
            best_model = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_predictor = {key: value.detach().cpu().clone() for key, value in predictor.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_model is None or best_predictor is None:
        raise RuntimeError("HGT training did not produce a checkpoint")
    model.load_state_dict(best_model)
    predictor.load_state_dict(best_predictor)
    selected = select_hgt_checkpoint(history)
    _require_export_ready(selected["validation"])
    metrics_path, _ = _training_metrics_paths(args.checkpoint)
    metrics_payload = {
        "schema_version": 1,
        "scenario": manifest["scenario"],
        "seed": manifest["seed"],
        "manifest_hash": manifest["manifest_hash"],
        "selected_epoch": selected["epoch"],
        "selection": "highest_validation_macro_relation_ap_then_lower_bce_then_earlier_epoch",
        "graph_provenance": graph_provenance,
        "history": history,
    }
    metrics_hash = _write_training_metrics(metrics_path, metrics_payload)
    source_hash = hashlib.sha256(args.primekg.read_bytes()).hexdigest()
    checkpoint_hash = save_hgt_checkpoint(
        model,
        args.checkpoint,
        source_sha256=source_hash,
        training_config={
            "scenario": manifest["scenario"],
            "seed": manifest["seed"],
            "manifest_hash": manifest["manifest_hash"],
            **graph_provenance,
            "best_validation_loss": selected["validation"]["global"]["bce"],
            "best_validation_macro_relation_ap": selected["validation"]["macro_relation_ap"],
            "max_train_edges_per_relation": args.max_train_edges_per_relation,
            "max_validation_edges_per_relation": args.max_validation_edges_per_relation,
            "hgt_training_metrics_path": str(metrics_path),
            "hgt_training_metrics_sha256": metrics_hash,
            "max_train_edges_per_relation": args.max_train_edges_per_relation,
            "max_validation_edges_per_relation": args.max_validation_edges_per_relation,
            "morgan_sha256": hashlib.sha256(args.morgan.read_bytes()).hexdigest(),
        },
        predictor=predictor,
    )
    tokens, available = _export_tokens(
        model.to(args.device),
        train_data,
        required_drugs,
        prepared.node_ids["drug"],
        neighbors=args.num_neighbors,
        batch_size=args.batch_size,
        device=args.device,
    )
    artifact = TokenFeatureArtifact.write(
        args.output,
        required_drugs,
        tokens,
        np.zeros(tokens.shape[:2], dtype=bool),
        available,
        producer="primekg_hgt",
        producer_config={
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "heads": args.heads,
            "scenario": manifest["scenario"],
            "seed": manifest["seed"],
            "manifest_hash": manifest["manifest_hash"],
            **graph_provenance,
            "hgt_training_metrics_path": str(metrics_path),
            "hgt_training_metrics_sha256": metrics_hash,
        },
        source_sha256=source_hash,
        checkpoint_sha256=checkpoint_hash,
    )
    _validate_hgt_message_graph_artifact(artifact, message)
    print(
        f"Wrote HGT checkpoint {args.checkpoint} and {len(artifact.drug_ids)} "
        f"benchmark token rows to {args.output}"
    )


if __name__ == "__main__":
    main()

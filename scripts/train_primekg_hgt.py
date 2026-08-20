"""Train a typed, leakage-safe PrimeKG HGT and export benchmark drug tokens."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import sys
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data.primekg_typed_graph import (
    PreparedTypedPrimeKG,
    build_leakage_safe_edge_partitions,
    prepare_typed_primekg,
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
) -> tuple[torch.Tensor, torch.Tensor]:
    negatives = sample_bipartite_negatives(
        all_positives,
        num_source_nodes=source_count,
        num_destination_nodes=destination_count,
        count=positives.shape[1],
        seed=seed,
    )
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
    max_positive_edges=None,
    phase="train",
    optimizer=None,
) -> float:
    from torch_geometric.loader import LinkNeighborLoader

    total = 0.0
    count = 0
    relations = sorted(supervision.items())
    for relation_number, (edge_type, positives) in enumerate(relations):
        source_type, _, destination_type = edge_type
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
        edge_label_index, edge_label = _labels(
            positives,
            all_edges[edge_type],
            len(node_ids[source_type]),
            len(node_ids[destination_type]),
            seed + relation_number,
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
            logits = predictor(output, edge_type, batch[edge_type].edge_label_index)
            loss = F.binary_cross_entropy_with_logits(logits, batch[edge_type].edge_label)
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            total += float(loss.detach()) * int(logits.numel())
            count += int(logits.numel())
    if count == 0:
        raise RuntimeError("HGT edge partitions contain no supervised examples")
    return total / count


def _export_tokens(
    model,
    full_data,
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
            full_data,
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
    best_loss = math.inf
    best_model = None
    best_predictor = None
    stale = 0
    for epoch in range(args.epochs):
        model.train()
        predictor.train()
        train_loss = _relation_loss(
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
            max_positive_edges=args.max_train_edges_per_relation,
            phase="train",
            optimizer=optimizer,
        )
        model.eval()
        predictor.eval()
        with torch.no_grad():
            validation_loss = _relation_loss(
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
                max_positive_edges=args.max_validation_edges_per_relation,
                phase="validation",
            )
        print(
            f"epoch={epoch + 1}/{args.epochs} train_loss={train_loss:.6f} "
            f"validation_loss={validation_loss:.6f}"
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
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
    source_hash = hashlib.sha256(args.primekg.read_bytes()).hexdigest()
    checkpoint_hash = save_hgt_checkpoint(
        model,
        args.checkpoint,
        source_sha256=source_hash,
        training_config={
            "scenario": manifest["scenario"],
            "seed": manifest["seed"],
            "manifest_hash": manifest["manifest_hash"],
            "excluded_cold_drug_count": len(excluded),
            "best_validation_loss": best_loss,
            "max_train_edges_per_relation": args.max_train_edges_per_relation,
            "max_validation_edges_per_relation": args.max_validation_edges_per_relation,
            "morgan_sha256": hashlib.sha256(args.morgan.read_bytes()).hexdigest(),
        },
        predictor=predictor,
    )
    full_x = _node_features(
        prepared,
        prepared.edge_index,
        args.morgan,
        args.drug_id_column,
        args.morgan_column,
    )
    full_data = _heterodata(full_x, prepared.edge_index)
    tokens, available = _export_tokens(
        model.to(args.device),
        full_data,
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
            "excluded_cold_drug_count": len(excluded),
            "max_train_edges_per_relation": args.max_train_edges_per_relation,
            "max_validation_edges_per_relation": args.max_validation_edges_per_relation,
            "cold_protocol": "label_and_encoder_train_inductive_safe_graph_at_export",
        },
        source_sha256=source_hash,
        checkpoint_sha256=checkpoint_hash,
    )
    print(
        f"Wrote HGT checkpoint {args.checkpoint} and {len(artifact.drug_ids)} "
        f"benchmark token rows to {args.output}"
    )


if __name__ == "__main__":
    main()

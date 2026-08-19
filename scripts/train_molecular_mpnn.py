"""Self-supervise the molecular MPNN with masked atom identity."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from src.models.molecular_mpnn import (
    MolecularMPNN,
    collate_molecular_graphs,
    featurize_smiles,
    masked_atom_loss,
    save_mpnn_checkpoint,
)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def _device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--token-count", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--mask-probability", type=float, default=0.15)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        raise ValueError("epochs, batch-size, and patience must be positive")
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("validation-fraction must be in (0, 0.5)")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    frame = _read(args.input)
    if args.smiles_column not in frame:
        raise ValueError(f"drug input is missing {args.smiles_column}")
    graphs = [featurize_smiles(value) for value in frame[args.smiles_column].tolist()]
    if len(graphs) < 2:
        raise ValueError("at least two valid molecules are required")
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(graphs), generator=generator).tolist()
    validation_count = max(1, int(round(len(graphs) * args.validation_fraction)))
    validation_indices = order[:validation_count]
    train_indices = order[validation_count:]
    device = _device(args.device)
    model = MolecularMPNN(args.hidden_dim, args.layers, args.token_count).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_state = None
    best_loss = math.inf
    stale = 0
    for epoch in range(args.epochs):
        model.train()
        shuffled = torch.tensor(train_indices)[
            torch.randperm(len(train_indices), generator=torch.Generator().manual_seed(args.seed + epoch))
        ].tolist()
        train_total = 0.0
        batches = 0
        for batch_number, start in enumerate(range(0, len(shuffled), args.batch_size)):
            batch = collate_molecular_graphs(
                [graphs[index] for index in shuffled[start : start + args.batch_size]]
            )
            optimizer.zero_grad(set_to_none=True)
            loss = masked_atom_loss(
                model,
                batch,
                mask_probability=args.mask_probability,
                generator_seed=args.seed + epoch * 1_000_003 + batch_number,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_total += float(loss.detach())
            batches += 1
        model.eval()
        validation_batch = collate_molecular_graphs(
            [graphs[index] for index in validation_indices]
        )
        with torch.no_grad():
            validation_loss = float(
                masked_atom_loss(
                    model,
                    validation_batch,
                    mask_probability=args.mask_probability,
                    generator_seed=args.seed + 999_983,
                )
            )
        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"train_loss={train_total / max(batches, 1):.6f} "
            f"validation_loss={validation_loss:.6f}"
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("MPNN training did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    checksum = save_mpnn_checkpoint(
        model,
        args.checkpoint,
        source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        training_config={
            "objective": "masked_atom_identity",
            "seed": args.seed,
            "epochs_requested": args.epochs,
            "best_validation_loss": best_loss,
            "validation_fraction": args.validation_fraction,
        },
    )
    print(f"Wrote MPNN checkpoint {args.checkpoint} sha256={checksum}")


if __name__ == "__main__":
    main()

"""Export fixed molecular-MPNN atom tokens for every drug."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.models.molecular_mpnn import export_mpnn_token_artifact, load_mpnn_checkpoint


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drug-id-column", default="drugbank_id")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = _read(args.input)
    missing = sorted({args.drug_id_column, args.smiles_column} - set(frame.columns))
    if missing:
        raise ValueError(f"drug input is missing columns: {', '.join(missing)}")
    if frame[args.drug_id_column].astype(str).duplicated().any():
        raise ValueError("drug input contains duplicate drug IDs")
    model, checkpoint_metadata = load_mpnn_checkpoint(
        args.checkpoint, expected_sha256=args.checkpoint_sha256
    )
    training_config = checkpoint_metadata.get("training_config", {})
    upstream_validation_loss = (
        training_config.get("best_validation_loss")
        if isinstance(training_config, dict)
        else None
    )
    checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    artifact = export_mpnn_token_artifact(
        model,
        drug_ids=frame[args.drug_id_column].astype(str).tolist(),
        smiles=frame[args.smiles_column].tolist(),
        output_path=args.output,
        source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        checkpoint_sha256=checkpoint_sha256,
        batch_size=args.batch_size,
        device=args.device,
        upstream_validation_loss=upstream_validation_loss,
    )
    print(f"Wrote {len(artifact.drug_ids)} MPNN rows to {args.output}")


if __name__ == "__main__":
    main()

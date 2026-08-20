"""Extract pinned MolFormer token embeddings for every drug."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.features.molformer_token_producer import (
    DEFAULT_MODEL_ID,
    MolFormerTokenProducer,
    export_molformer_token_artifact,
)


def _table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("drug input must be CSV or Parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", required=True, help="Pinned 40-character HF commit hash")
    parser.add_argument("--allow-remote-code", action="store_true")
    parser.add_argument("--drug-id-column", default="drugbank_id")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--output-token-count", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = _table(args.input)
    missing = sorted({args.drug_id_column, args.smiles_column} - set(frame.columns))
    if missing:
        raise ValueError(f"drug input is missing columns: {', '.join(missing)}")
    if frame[args.drug_id_column].astype(str).duplicated().any():
        raise ValueError("drug input contains duplicate drug IDs")
    producer = MolFormerTokenProducer.from_pretrained(
        model_id=args.model_id,
        revision=args.revision,
        allow_remote_code=args.allow_remote_code,
        max_length=args.max_length,
        output_token_count=args.output_token_count,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    artifact = export_molformer_token_artifact(
        producer,
        drug_ids=frame[args.drug_id_column].astype(str).tolist(),
        smiles=frame[args.smiles_column].tolist(),
        output_path=args.output,
        source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        batch_size=args.batch_size,
    )
    print(f"Wrote {len(artifact.drug_ids)} MolFormer rows to {args.output}")


if __name__ == "__main__":
    main()

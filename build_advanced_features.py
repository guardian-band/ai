"""Join Morgan, MolFormer, MPNN, and HGT outputs for one benchmark manifest."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.advanced_feature_assembler import assemble_multimodal_features
from src.features.token_feature_artifact import TokenFeatureArtifact
from src.training.engine import verify_manifest


def _resolve(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--morgan", type=Path, required=True)
    parser.add_argument("--molformer", type=Path, required=True)
    parser.add_argument("--mpnn", type=Path, required=True)
    parser.add_argument("--kg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drug-id-column", default="drugbank_id")
    parser.add_argument("--morgan-column", default="morgan_fingerprint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = verify_manifest(str(manifest_path))
    pairs_path = _resolve(manifest_path, str(manifest["pairs_path"]))
    pairs = pd.read_parquet(pairs_path)
    required = sorted(set(pairs["drug_a"].astype(str)) | set(pairs["drug_b"].astype(str)))
    morgan_frame = pd.read_parquet(args.morgan)
    missing = sorted({args.drug_id_column, args.morgan_column} - set(morgan_frame.columns))
    if missing:
        raise ValueError(f"Morgan artifact is missing columns: {', '.join(missing)}")
    if morgan_frame[args.drug_id_column].astype(str).duplicated().any():
        raise ValueError("Morgan artifact contains duplicate drug IDs")
    try:
        morgan = np.stack(morgan_frame[args.morgan_column].map(np.asarray).tolist()).astype(
            np.float32, copy=False
        )
    except ValueError as exc:
        raise ValueError("Morgan artifact rows must have one fixed feature dimension") from exc
    artifact = assemble_multimodal_features(
        args.output,
        required_drug_ids=required,
        morgan_drug_ids=morgan_frame[args.drug_id_column].astype(str).tolist(),
        morgan=morgan,
        molformer=TokenFeatureArtifact.load(args.molformer),
        mpnn=TokenFeatureArtifact.load(args.mpnn),
        kg=TokenFeatureArtifact.load(args.kg),
        morgan_provenance_hash=hashlib.sha256(args.morgan.read_bytes()).hexdigest(),
        manifest_compatibility={
            "benchmark_id": manifest["benchmark_id"],
            "scenario": manifest["scenario"],
            "seed": manifest["seed"],
            "manifest_hash": manifest["manifest_hash"],
        },
    )
    print(f"Wrote {len(artifact.drug_ids)} complete multimodal rows to {args.output}")


if __name__ == "__main__":
    main()

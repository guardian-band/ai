"""Manifest-backed benchmark examples for the executable experiment path.

The benchmark builder emits three small, portable artifacts referenced by a
manifest: ``pairs.parquet``, ``triples.parquet``, and ``labels.json``.  This
module validates those artifacts and exposes one deterministic example per
pair.  The token-shaped inputs are deliberately documented ID features; they
are a temporary adapter for the current model interface and are not
ChemBERTa representations.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np


PAIR_COLUMNS = {
    "pair_id",
    "drug_a",
    "drug_b",
    "split",
    "scenario",
    "observation_status",
    "source_positive_pair_id",
}
TRIPLE_COLUMNS = {"pair_id", "drug_a", "drug_b", "label_cui", "observation_status"}
VALID_SPLITS = {"train", "validation", "test"}
POSITIVE_STATUSES = {"observed_positive", "positive", "known_positive"}


def _resolve_artifact_path(manifest_path: Path, manifest: Mapping[str, Any], key: str) -> Path:
    raw_path = manifest.get(key)
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"Manifest must define a non-empty {key}")
    path = Path(raw_path)
    return path if path.is_absolute() else manifest_path.parent / path


def _read_json_labels(labels_path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    try:
        payload = json.loads(labels_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read labels.json at {labels_path}: {exc}") from exc

    records = payload.get("labels") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError("labels.json must contain a non-empty 'labels' list")

    labels: list[dict[str, Any]] = []
    cui_to_index: dict[str, int] = {}
    for expected_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("labels.json labels must be objects")
        index = record.get("index")
        cui = record.get("cui")
        if isinstance(index, bool) or not isinstance(index, int) or index != expected_index:
            raise ValueError("labels.json indices must be contiguous and ordered from zero")
        if not isinstance(cui, str) or not cui:
            raise ValueError("labels.json each label must contain a non-empty cui")
        if cui in cui_to_index:
            raise ValueError("labels.json cui values must be unique")
        labels.append(dict(record))
        cui_to_index[cui] = index
    return labels, cui_to_index


def _read_parquet(path: Path, required_columns: set[str], artifact_name: str) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError, ImportError) as exc:
        raise ValueError(f"Unable to read {artifact_name} at {path}: {exc}") from exc
    missing = required_columns.difference(frame.columns)
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(f"{artifact_name} missing required columns: {missing_names}")
    return frame


def _normalize_status(value: Any) -> str:
    return str(value).strip().lower()


def _load_drug_features(parquet_path: Path) -> dict[str, torch.Tensor]:
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as exc:
        raise ValueError(f"Unable to read drug features at {parquet_path}: {exc}") from exc
    
    if "drugbank_id" not in df.columns or "morgan_fingerprint" not in df.columns:
        raise ValueError("Parquet must contain drugbank_id and morgan_fingerprint columns")
        
    features = {}
    for row in df.itertuples(index=False):
        fp = np.array(row.morgan_fingerprint, dtype=np.float32)
        # Shape (1, feature_dim) to mimic token sequence of length 1
        features[str(row.drugbank_id)] = torch.tensor(fp).unsqueeze(0)
    return features


class ManifestPolypharmacyDataset(Dataset):
    """Validated, deterministic one-row-per-pair benchmark dataset."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        labels: list[dict[str, Any]],
        drug_features_path: Path | str | None = None,
    ):
        if not drug_features_path:
            raise ValueError("drug_features_path is required")
        
        self.labels = labels
        self.drug_features = _load_drug_features(Path(drug_features_path))
        
        # Filter records that lack features
        valid_records = []
        for r in records:
            if r["drug_a"] in self.drug_features and r["drug_b"] in self.drug_features:
                valid_records.append(r)
        self.records = valid_records
        
        # We enforce token length 1 for our flat vectors to work with _masked_mean
        self.token_length = 1

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        manifest: Mapping[str, Any],
        split: str,
        *,
        drug_features_path: Path | str | None = None,
    ) -> "ManifestPolypharmacyDataset":
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}")
        manifest_path = Path(manifest_path).resolve()
        scenario = manifest.get("scenario")
        if not isinstance(scenario, str) or not scenario:
            raise ValueError("Manifest must define a non-empty scenario")

        pairs_path = _resolve_artifact_path(manifest_path, manifest, "pairs_path")
        triples_path = _resolve_artifact_path(manifest_path, manifest, "triples_path")
        labels_path = _resolve_artifact_path(manifest_path, manifest, "labels_path")
        labels, cui_to_index = _read_json_labels(labels_path)
        pairs = _read_parquet(pairs_path, PAIR_COLUMNS, "pairs.parquet")
        triples = _read_parquet(triples_path, TRIPLE_COLUMNS, "triples.parquet")

        if pairs["pair_id"].isna().any() or pairs["pair_id"].duplicated().any():
            raise ValueError("pair_id values must be unique and non-null")
        if pairs["scenario"].map(_normalize_status).ne(_normalize_status(scenario)).any():
            raise ValueError("scenario values must match manifest scenario")
        if pairs["split"].isna().any() or not set(pairs["split"].unique()).issubset(VALID_SPLITS):
            raise ValueError(f"split values must be one of {sorted(VALID_SPLITS)}")
        if pairs[["drug_a", "drug_b"]].isna().any().any():
            raise ValueError("pairs.parquet drug_a and drug_b values must be non-null")

        pair_by_id = pairs.set_index("pair_id", drop=False)
        source_pair_ids = pairs["source_positive_pair_id"].dropna()
        unknown_source_ids = set(source_pair_ids) - set(pair_by_id.index)
        if unknown_source_ids:
            raise ValueError("source_positive_pair_id references unknown pair_id")
        if triples[sorted(TRIPLE_COLUMNS)].isna().any().any():
            raise ValueError("triples.parquet required values must be non-null")
        unknown_pair_ids = set(triples["pair_id"].dropna()) - set(pair_by_id.index)
        if unknown_pair_ids:
            raise ValueError("triples.parquet references unknown pair_id")
        for row in triples.itertuples(index=False):
            pair = pair_by_id.loc[row.pair_id]
            if str(row.drug_a) != str(pair.drug_a) or str(row.drug_b) != str(pair.drug_b):
                raise ValueError("triples.parquet drug IDs must match their pair_id")
        unknown_cuis = set(triples["label_cui"].dropna()) - set(cui_to_index)
        if unknown_cuis:
            raise ValueError("label_cui values must exist in labels.json")

        selected_pairs = pairs[pairs["split"] == split].sort_values("pair_id").reset_index(drop=True)
        selected_ids = set(selected_pairs["pair_id"])
        selected_triples = triples[triples["pair_id"].isin(selected_ids)]
        labels_by_pair: dict[Any, list[str]] = {pair_id: [] for pair_id in selected_ids}
        for row in selected_triples.itertuples(index=False):
            if _normalize_status(row.observation_status) in POSITIVE_STATUSES:
                labels_by_pair[row.pair_id].append(str(row.label_cui))

        records: list[dict[str, Any]] = []
        for pair in selected_pairs.itertuples(index=False):
            pair_id = pair.pair_id
            label_vector = [0.0] * len(labels)
            for cui in labels_by_pair[pair_id]:
                label_vector[cui_to_index[cui]] = 1.0
            is_positive = _normalize_status(pair.observation_status) in POSITIVE_STATUSES
            is_positive = is_positive or bool(label_vector.count(1.0))
            records.append(
                {
                    "pair_id": pair_id,
                    "drug_a": str(pair.drug_a),
                    "drug_b": str(pair.drug_b),
                    "split": pair.split,
                    "labels": label_vector,
                    "is_observed_positive": is_positive,
                }
            )
        return cls(records, labels, drug_features_path=drug_features_path)

    def __len__(self) -> int:
        return len(self.records)

    def _features_for(self, drug_id: str) -> torch.Tensor:
        return self.drug_features[drug_id]

    def __getitem__(self, index: int):
        record = self.records[index]
        mask_a = torch.zeros(self.token_length, dtype=torch.bool)
        mask_b = torch.zeros(self.token_length, dtype=torch.bool)
        labels = torch.tensor(record["labels"], dtype=torch.float32)
        return (
            self._features_for(record["drug_a"]).clone(),
            self._features_for(record["drug_b"]).clone(),
            mask_a,
            mask_b,
            labels,
            bool(record["is_observed_positive"]),
        )

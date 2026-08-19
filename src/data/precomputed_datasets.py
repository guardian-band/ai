"""Manifest pair datasets backed by validated precomputed feature artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.manifest_dataset import ManifestRecords, load_manifest_records
from src.features.cached_token_artifact import CachedTokenArtifact
from src.features.multimodal_feature_artifact import MultimodalFeatureArtifact
from src.models.hierarchy import HierarchyMapping, load_hierarchy_mapping


def _resolve_hierarchy(path: str | Path, labels: tuple[Mapping[str, Any], ...]) -> HierarchyMapping:
    cuis = [str(label["cui"]) for label in labels]
    return load_hierarchy_mapping(path, selected_specific_cuis=cuis)


def _organ_targets(
    specific_targets: torch.Tensor,
    labels: tuple[Mapping[str, Any], ...],
    hierarchy: HierarchyMapping,
) -> torch.Tensor:
    organ = torch.zeros(len(hierarchy.organ_order), dtype=torch.float32)
    for index, label in enumerate(labels):
        if float(specific_targets[index]) > 0:
            organ[hierarchy.specific_to_organ[str(label["cui"])]] = 1.0
    return organ


def _validated_records(
    manifest_path: str | Path, manifest: Mapping[str, Any], split: str, hierarchy_path: str | Path
) -> tuple[ManifestRecords, HierarchyMapping]:
    parsed = load_manifest_records(manifest_path, manifest, split)
    hierarchy = _resolve_hierarchy(hierarchy_path, parsed.labels)
    return parsed, hierarchy


def _missing_drugs(records: tuple[Mapping[str, Any], ...], drug_ids: tuple[str, ...]) -> list[str]:
    required = {str(record["drug_a"]) for record in records} | {str(record["drug_b"]) for record in records}
    return sorted(required - set(drug_ids))


def validate_drug_coverage(
    parsed: ManifestRecords, drug_ids: tuple[str, ...], *, split: str
) -> None:
    """Validate a split's complete drug join without constructing a dataset.

    The preflight path uses this for the test split so it can validate every
    manifest/artifact join while still deferring test dataset/DataLoader
    construction until validation has been frozen.
    """

    missing = _missing_drugs(parsed.records, drug_ids)
    if missing:
        raise ValueError(f"missing drug IDs in {split} ({len(missing)}): {', '.join(missing)}")


class MultimodalPairDataset(Dataset):
    """One validated manifest pair per item using multimodal drug features."""

    def __init__(self, parsed: ManifestRecords, feature_artifact: MultimodalFeatureArtifact, hierarchy: HierarchyMapping):
        missing = _missing_drugs(parsed.records, feature_artifact.drug_ids)
        if missing:
            raise ValueError(f"missing drug IDs ({len(missing)}): {', '.join(missing)}")
        self.records = [dict(record, labels=list(record["labels"])) for record in parsed.records]
        self.labels = [dict(label) for label in parsed.labels]
        self.feature_artifact = feature_artifact
        self.hierarchy = hierarchy

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        manifest: Mapping[str, Any],
        split: str,
        feature_artifact: MultimodalFeatureArtifact | str | Path,
        hierarchy_path: str | Path,
    ) -> "MultimodalPairDataset":
        artifact = (
            MultimodalFeatureArtifact.load(feature_artifact)
            if isinstance(feature_artifact, (str, Path))
            else feature_artifact
        )
        if not isinstance(artifact, MultimodalFeatureArtifact):
            raise TypeError("feature_artifact must be a validated MultimodalFeatureArtifact")
        parsed, hierarchy = _validated_records(manifest_path, manifest, split, hierarchy_path)
        return cls(parsed, artifact, hierarchy)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        specific = torch.tensor(record["labels"], dtype=torch.float32)
        organ = _organ_targets(specific, tuple(self.labels), self.hierarchy)
        drug_a = self.feature_artifact.lookup(record["drug_a"])
        drug_b = self.feature_artifact.lookup(record["drug_b"])

        def tensors(values: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
            return {
                key: torch.from_numpy(value.copy()) if isinstance(value, np.ndarray) and value.ndim else torch.tensor(value.item())
                for key, value in values.items()
            }

        return (
            tensors(drug_a),
            tensors(drug_b),
            organ,
            specific,
            str(record["pair_id"]),
            bool(record["is_observed_positive"]),
        )


class StudentPairDataset(Dataset):
    """One validated manifest pair per item using cached student tokens."""

    def __init__(self, parsed: ManifestRecords, cached_artifact: CachedTokenArtifact, hierarchy: HierarchyMapping):
        missing = _missing_drugs(parsed.records, cached_artifact.drug_ids)
        if missing:
            raise ValueError(f"missing drug IDs ({len(missing)}): {', '.join(missing)}")
        self.records = [dict(record, labels=list(record["labels"])) for record in parsed.records]
        self.labels = [dict(label) for label in parsed.labels]
        self.cached_artifact = cached_artifact
        self.hierarchy = hierarchy

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        manifest: Mapping[str, Any],
        split: str,
        cached_artifact: CachedTokenArtifact | str | Path,
        hierarchy_path: str | Path,
    ) -> "StudentPairDataset":
        artifact = (
            CachedTokenArtifact.load(cached_artifact)
            if isinstance(cached_artifact, (str, Path))
            else cached_artifact
        )
        if not isinstance(artifact, CachedTokenArtifact):
            raise TypeError("cached_artifact must be a validated CachedTokenArtifact")
        parsed, hierarchy = _validated_records(manifest_path, manifest, split, hierarchy_path)
        return cls(parsed, artifact, hierarchy)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        specific = torch.tensor(record["labels"], dtype=torch.float32)
        organ = _organ_targets(specific, tuple(self.labels), self.hierarchy)
        tokens_a, available_a = self.cached_artifact.lookup(record["drug_a"])
        tokens_b, available_b = self.cached_artifact.lookup(record["drug_b"])
        return (
            torch.from_numpy(tokens_a),
            torch.from_numpy(tokens_b),
            torch.from_numpy(available_a),
            torch.from_numpy(available_b),
            organ,
            specific,
            str(record["pair_id"]),
            bool(record["is_observed_positive"]),
        )

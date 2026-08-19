"""Deterministic join of advanced-model upstream feature artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from src.features.multimodal_feature_artifact import MultimodalFeatureArtifact
from src.features.token_feature_artifact import TokenFeatureArtifact


def _indices(required: tuple[str, ...], available: Sequence[str], name: str) -> np.ndarray:
    mapping = {drug_id: index for index, drug_id in enumerate(available)}
    if len(mapping) != len(available):
        raise ValueError(f"{name} contains duplicate drug IDs")
    missing = sorted(set(required) - set(mapping))
    if missing:
        raise ValueError(f"{name} is missing required drug IDs: {', '.join(missing)}")
    return np.asarray([mapping[drug_id] for drug_id in required], dtype=np.int64)


def assemble_multimodal_features(
    output_path: str | Path,
    *,
    required_drug_ids: Sequence[str],
    morgan_drug_ids: Sequence[str],
    morgan: np.ndarray,
    molformer: TokenFeatureArtifact,
    mpnn: TokenFeatureArtifact,
    kg: TokenFeatureArtifact,
    morgan_provenance_hash: str,
    manifest_compatibility: Mapping[str, object],
) -> MultimodalFeatureArtifact:
    """Fail-closed exact join for one benchmark scenario/seed manifest."""

    raw_required = list(required_drug_ids)
    if any(not isinstance(item, str) or not item.strip() for item in raw_required):
        raise ValueError("required drug IDs must be non-empty strings")
    if len(set(raw_required)) != len(raw_required):
        raise ValueError("required drug IDs contain duplicates")
    required = tuple(sorted(raw_required))
    if not required:
        raise ValueError("at least one required drug ID is needed")
    morgan_ids = list(morgan_drug_ids)
    morgan_values = np.asarray(morgan)
    if (
        morgan_values.dtype != np.float32
        or morgan_values.ndim != 2
        or morgan_values.shape[0] != len(morgan_ids)
        or not np.isfinite(morgan_values).all()
    ):
        raise ValueError("morgan must be finite float32 with one row per Morgan drug ID")
    morgan_index = _indices(required, morgan_ids, "morgan")
    molformer_index = _indices(required, molformer.drug_ids, "molformer")
    mpnn_index = _indices(required, mpnn.drug_ids, "mpnn")
    kg_index = _indices(required, kg.drug_ids, "kg")
    count = len(required)
    return MultimodalFeatureArtifact.write(
        output_path,
        drug_ids=required,
        morgan=np.ascontiguousarray(morgan_values[morgan_index]),
        molformer_tokens=np.ascontiguousarray(molformer.tokens[molformer_index]),
        mpnn_tokens=np.ascontiguousarray(mpnn.tokens[mpnn_index]),
        kg_tokens=np.ascontiguousarray(kg.tokens[kg_index]),
        morgan_available=np.ones(count, dtype=bool),
        molformer_available=np.ascontiguousarray(molformer.available[molformer_index]),
        mpnn_available=np.ascontiguousarray(mpnn.available[mpnn_index]),
        kg_available=np.ascontiguousarray(kg.available[kg_index]),
        molformer_padding_mask=np.ascontiguousarray(
            molformer.padding_mask[molformer_index]
        ),
        mpnn_padding_mask=np.ascontiguousarray(mpnn.padding_mask[mpnn_index]),
        kg_padding_mask=np.ascontiguousarray(kg.padding_mask[kg_index]),
        morgan_provenance_hash=morgan_provenance_hash,
        molformer_provenance_hash=molformer.provenance_sha256,
        mpnn_provenance_hash=mpnn.provenance_sha256,
        kg_provenance_hash=kg.provenance_sha256,
        manifest_compatibility=manifest_compatibility,
    )


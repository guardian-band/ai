import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.manifest_dataset import load_manifest_records
from src.data.precomputed_datasets import MultimodalPairDataset, StudentPairDataset
from src.features.cached_token_artifact import CachedTokenArtifact
from src.features.multimodal_feature_artifact import MultimodalFeatureArtifact


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64


def _feature_arrays(count, *, morgan=None):
    return {
        "morgan": np.ones((count, 4), dtype=np.float32) if morgan is None else morgan,
        "molformer_tokens": np.ones((count, 2, 3), dtype=np.float32),
        "mpnn_tokens": np.ones((count, 3, 5), dtype=np.float32),
        "kg_tokens": np.ones((count, 1, 2), dtype=np.float32),
        "morgan_available": np.ones(count, dtype=bool),
        "molformer_available": np.ones(count, dtype=bool),
        "mpnn_available": np.ones(count, dtype=bool),
        "kg_available": np.ones(count, dtype=bool),
        "molformer_padding_mask": np.zeros((count, 2), dtype=bool),
        "mpnn_padding_mask": np.zeros((count, 3), dtype=bool),
        "kg_padding_mask": np.zeros((count, 1), dtype=bool),
    }


def _provenance(**overrides):
    values = {
        "morgan_provenance_hash": HASH_A,
        "molformer_provenance_hash": HASH_B,
        "mpnn_provenance_hash": HASH_C,
        "kg_provenance_hash": HASH_D,
    }
    values.update(overrides)
    return values


def test_multimodal_artifact_keeps_molformer_and_mpnn_as_separate_modalities(tmp_path):
    artifact = MultimodalFeatureArtifact.write(
        tmp_path / "advanced.npz",
        drug_ids=["D1"],
        morgan=np.ones((1, 4), dtype=np.float32),
        molformer_tokens=np.ones((1, 3, 6), dtype=np.float32),
        mpnn_tokens=np.ones((1, 2, 5), dtype=np.float32),
        kg_tokens=np.ones((1, 4, 7), dtype=np.float32),
        morgan_available=np.ones(1, dtype=bool),
        molformer_available=np.ones(1, dtype=bool),
        mpnn_available=np.ones(1, dtype=bool),
        kg_available=np.ones(1, dtype=bool),
        molformer_padding_mask=np.zeros((1, 3), dtype=bool),
        mpnn_padding_mask=np.zeros((1, 2), dtype=bool),
        kg_padding_mask=np.zeros((1, 4), dtype=bool),
        morgan_provenance_hash=HASH_A,
        molformer_provenance_hash=HASH_B,
        mpnn_provenance_hash=HASH_C,
        kg_provenance_hash=HASH_D,
        manifest_compatibility={
            "benchmark_id": "fixture",
            "scenario": "cold_1",
            "seed": 1,
            "manifest_hash": "manifest",
        },
    )
    row = artifact.lookup("D1")
    assert row["molformer_tokens"].shape == (3, 6)
    assert row["mpnn_tokens"].shape == (2, 5)
    assert artifact.metadata["schema_version"] == 2


@pytest.fixture
def precomputed_fixture(tmp_path):
    pairs = pd.DataFrame(
        [
            {"pair_id": "p1", "drug_a": "D1", "drug_b": "D2", "split": "train", "scenario": "warm_pair", "observation_status": "observed_positive", "source_positive_pair_id": None},
            {"pair_id": "p2", "drug_a": "D2", "drug_b": "D3", "split": "validation", "scenario": "warm_pair", "observation_status": "unlabeled", "source_positive_pair_id": None},
            {"pair_id": "p3", "drug_a": "D1", "drug_b": "D3", "split": "test", "scenario": "warm_pair", "observation_status": "unlabeled", "source_positive_pair_id": None},
        ]
    )
    triples = pd.DataFrame(
        [{"pair_id": "p1", "drug_a": "D1", "drug_b": "D2", "label_cui": "C1", "observation_status": "observed_positive"}]
    )
    pairs.to_parquet(tmp_path / "pairs.parquet", index=False)
    triples.to_parquet(tmp_path / "triples.parquet", index=False)
    (tmp_path / "labels.json").write_text(json.dumps({"labels": [{"index": 0, "cui": "C1"}, {"index": 1, "cui": "C2"}]}))
    manifest_body = {"benchmark_id": "fixture", "seed": 1, "scenario": "warm_pair", "pairs_path": "pairs.parquet", "triples_path": "triples.parquet", "labels_path": "labels.json"}
    manifest_hash = hashlib.sha256(json.dumps(manifest_body, sort_keys=True).encode()).hexdigest()[:12]
    manifest = dict(manifest_body, manifest_hash=manifest_hash)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    ids = ["D1", "D2", "D3"]
    feature = MultimodalFeatureArtifact.write(
        tmp_path / "features.npz",
        drug_ids=ids,
        **_feature_arrays(3),
        **_provenance(),
        manifest_compatibility={"benchmark_id": "fixture", "scenario": "warm_pair", "seed": 1, "manifest_hash": manifest_hash},
    )
    cached = CachedTokenArtifact.write(
        tmp_path / "cached.npz",
        ids,
        np.ones((3, 3, 6), dtype=np.float32),
        np.ones((3, 3), dtype=bool),
        teacher_provenance_hash=HASH_A,
        teacher_checkpoint_hash=HASH_A,
        teacher_config_hash=HASH_B,
        teacher_selection_hash=HASH_E,
        teacher_selected_mode="baseline",
        modality_provenance_hash=HASH_C,
    )
    hierarchy = {"schema_version": 1, "organ_order": ["O1"], "mappings": [{"specific_cui": "C1", "organ_index": 0}, {"specific_cui": "C2", "organ_index": 0}]}
    hierarchy_path = tmp_path / "hierarchy.json"
    hierarchy_path.write_text(json.dumps(hierarchy))
    return manifest_path, manifest, feature, cached, hierarchy_path


def test_shared_manifest_reader_returns_ordered_read_only_records(precomputed_fixture):
    manifest_path, manifest, *_ = precomputed_fixture
    parsed = load_manifest_records(manifest_path, manifest, split="validation")
    assert tuple(record["pair_id"] for record in parsed.records) == ("p2",)
    with pytest.raises(TypeError):
        parsed.records[0]["drug_a"] = "D9"


def test_multimodal_and_student_pair_datasets_join_without_filtering(precomputed_fixture):
    manifest_path, manifest, feature, cached, hierarchy_path = precomputed_fixture
    multimodal = MultimodalPairDataset.from_manifest(manifest_path, manifest, "train", feature, hierarchy_path)
    student = StudentPairDataset.from_manifest(manifest_path, manifest, "train", cached, hierarchy_path)
    assert len(multimodal) == len(student) == 1
    modality_a, modality_b, organ, specific, *_ = multimodal[0]
    tokens_a, tokens_b, _, _, student_organ, student_specific, *_ = student[0]
    assert modality_a["morgan"].shape == (4,)
    assert modality_b["kg_tokens"].shape == (1, 2)
    assert tokens_a.shape == (3, 6)
    assert torch.equal(organ, student_organ)
    assert torch.equal(specific, student_specific)


def test_multimodal_dataset_reports_all_missing_drugs(precomputed_fixture):
    manifest_path, manifest, feature, *_ = precomputed_fixture
    missing = MultimodalFeatureArtifact.write(
        manifest_path.parent / "missing.npz",
        drug_ids=["D1"],
        **_feature_arrays(1),
        **_provenance(),
        manifest_compatibility={"benchmark_id": "fixture", "scenario": "warm_pair", "seed": 1, "manifest_hash": manifest["manifest_hash"]},
    )
    with pytest.raises(ValueError, match="missing drug IDs.*D2"):
        MultimodalPairDataset.from_manifest(manifest_path, manifest, "train", missing, precomputed_fixture[-1])


def test_multimodal_feature_artifact_rejects_duplicate_nonfinite_and_bad_hash(tmp_path):
    kwargs = dict(
        **_provenance(),
        manifest_compatibility={"benchmark_id": "fixture", "scenario": "warm_pair", "seed": 1, "manifest_hash": "fixture_hash"},
    )
    with pytest.raises(ValueError, match="duplicate"):
        MultimodalFeatureArtifact.write(
            tmp_path / "duplicate.npz", drug_ids=["D1", "D1"],
            **_feature_arrays(2), **kwargs
        )
    with pytest.raises(ValueError, match="finite"):
        MultimodalFeatureArtifact.write(
            tmp_path / "nan.npz", drug_ids=["D1"],
            **_feature_arrays(1, morgan=np.full((1, 4), np.nan, dtype=np.float32)),
            **kwargs
        )
    with pytest.raises(ValueError, match="SHA-256"):
        MultimodalFeatureArtifact.write(
            tmp_path / "hash.npz", drug_ids=["D1"], **_feature_arrays(1),
            **_provenance(morgan_provenance_hash="bad"),
            manifest_compatibility={"benchmark_id": "fixture", "scenario": "warm_pair", "seed": 1, "manifest_hash": "fixture_hash"}
        )


def test_feature_metadata_and_arrays_are_immutable_after_validation(precomputed_fixture):
    _, _, feature, *_ = precomputed_fixture
    with pytest.raises(TypeError):
        feature.metadata["manifest_compatibility"]["benchmark_id"] = "tampered"
    with pytest.raises(ValueError):
        feature.morgan[0, 0] = 0.0


def test_feature_artifact_requires_complete_manifest_compatibility(tmp_path):
    with pytest.raises(ValueError, match="manifest_compatibility missing|required"):
        MultimodalFeatureArtifact.write(
            tmp_path / "incomplete.npz", drug_ids=["D1"],
            **_feature_arrays(1), **_provenance(),
            manifest_compatibility={"benchmark_id": "fixture"},
        )

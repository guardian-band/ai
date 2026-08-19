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
        ids,
        np.ones((3, 4), dtype=np.float32),
        np.ones((3, 2, 3), dtype=np.float32),
        np.ones((3, 1, 2), dtype=np.float32),
        np.ones(3, dtype=bool),
        np.ones(3, dtype=bool),
        np.ones(3, dtype=bool),
        np.zeros((3, 2), dtype=bool),
        np.zeros((3, 1), dtype=bool),
        morgan_provenance_hash=HASH_A,
        molecular_provenance_hash=HASH_B,
        kg_provenance_hash=HASH_C,
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
        ["D1"],
        np.ones((1, 4), dtype=np.float32),
        np.ones((1, 2, 3), dtype=np.float32),
        np.ones((1, 1, 2), dtype=np.float32),
        np.ones(1, dtype=bool), np.ones(1, dtype=bool), np.ones(1, dtype=bool),
        np.zeros((1, 2), dtype=bool), np.zeros((1, 1), dtype=bool),
        morgan_provenance_hash=HASH_A, molecular_provenance_hash=HASH_B, kg_provenance_hash=HASH_C,
        manifest_compatibility={"benchmark_id": "fixture", "scenario": "warm_pair", "seed": 1, "manifest_hash": manifest["manifest_hash"]},
    )
    with pytest.raises(ValueError, match="missing drug IDs.*D2"):
        MultimodalPairDataset.from_manifest(manifest_path, manifest, "train", missing, precomputed_fixture[-1])


def test_multimodal_feature_artifact_rejects_duplicate_nonfinite_and_bad_hash(tmp_path):
    kwargs = dict(
        morgan_provenance_hash=HASH_A,
        molecular_provenance_hash=HASH_B,
        kg_provenance_hash=HASH_C,
        manifest_compatibility={"benchmark_id": "fixture", "scenario": "warm_pair", "seed": 1, "manifest_hash": "fixture_hash"},
    )
    with pytest.raises(ValueError, match="duplicate"):
        MultimodalFeatureArtifact.write(
            tmp_path / "duplicate.npz", ["D1", "D1"], np.ones((2, 4), dtype=np.float32),
            np.ones((2, 2, 3), dtype=np.float32), np.ones((2, 1, 2), dtype=np.float32),
            np.ones(2, dtype=bool), np.ones(2, dtype=bool), np.ones(2, dtype=bool),
            np.zeros((2, 2), dtype=bool), np.zeros((2, 1), dtype=bool), **kwargs
        )
    with pytest.raises(ValueError, match="finite"):
        MultimodalFeatureArtifact.write(
            tmp_path / "nan.npz", ["D1"], np.full((1, 4), np.nan, dtype=np.float32),
            np.ones((1, 2, 3), dtype=np.float32), np.ones((1, 1, 2), dtype=np.float32),
            np.ones(1, dtype=bool), np.ones(1, dtype=bool), np.ones(1, dtype=bool),
            np.zeros((1, 2), dtype=bool), np.zeros((1, 1), dtype=bool), **kwargs
        )
    with pytest.raises(ValueError, match="SHA-256"):
        MultimodalFeatureArtifact.write(
            tmp_path / "hash.npz", ["D1"], np.ones((1, 4), dtype=np.float32),
            np.ones((1, 2, 3), dtype=np.float32), np.ones((1, 1, 2), dtype=np.float32),
            np.ones(1, dtype=bool), np.ones(1, dtype=bool), np.ones(1, dtype=bool),
            np.zeros((1, 2), dtype=bool), np.zeros((1, 1), dtype=bool),
            morgan_provenance_hash="bad", molecular_provenance_hash=HASH_B,
            kg_provenance_hash=HASH_C,
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
            tmp_path / "incomplete.npz", ["D1"],
            np.ones((1, 4), dtype=np.float32),
            np.ones((1, 2, 3), dtype=np.float32),
            np.ones((1, 1, 2), dtype=np.float32),
            np.ones(1, dtype=bool), np.ones(1, dtype=bool), np.ones(1, dtype=bool),
            np.zeros((1, 2), dtype=bool), np.zeros((1, 1), dtype=bool),
            morgan_provenance_hash=HASH_A, molecular_provenance_hash=HASH_B,
            kg_provenance_hash=HASH_C, manifest_compatibility={"benchmark_id": "fixture"},
        )

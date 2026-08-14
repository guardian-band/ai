import json

import pandas as pd
import pytest
import torch

from src.data.manifest_dataset import ManifestPolypharmacyDataset
from src.models.factory import PrevalenceModel


@pytest.fixture
def manifest_fixture(tmp_path):
    pairs = pd.DataFrame(
        [
            {
                "pair_id": "p_train_positive",
                "drug_a": "DB001",
                "drug_b": "DB002",
                "split": "train",
                "scenario": "warm_pair",
                "observation_status": "observed_positive",
                "source_positive_pair_id": None,
            },
            {
                "pair_id": "p_train_unlabeled",
                "drug_a": "DB001",
                "drug_b": "DB003",
                "split": "train",
                "scenario": "warm_pair",
                "observation_status": "unlabeled",
                "source_positive_pair_id": "p_train_positive",
            },
            {
                "pair_id": "p_validation",
                "drug_a": "DB002",
                "drug_b": "DB003",
                "split": "validation",
                "scenario": "warm_pair",
                "observation_status": "unlabeled",
                "source_positive_pair_id": None,
            },
            {
                "pair_id": "p_test",
                "drug_a": "DB001",
                "drug_b": "DB004",
                "split": "test",
                "scenario": "warm_pair",
                "observation_status": "observed_positive",
                "source_positive_pair_id": None,
            },
        ]
    )
    triples = pd.DataFrame(
        [
            {
                "pair_id": "p_train_positive",
                "drug_a": "DB001",
                "drug_b": "DB002",
                "label_cui": "C001",
                "observation_status": "observed_positive",
            },
            {
                "pair_id": "p_test",
                "drug_a": "DB001",
                "drug_b": "DB004",
                "label_cui": "C002",
                "observation_status": "observed_positive",
            },
        ]
    )
    labels = {
        "labels": [
            {"index": 0, "cui": "C001"},
            {"index": 1, "cui": "C002"},
        ]
    }
    pairs_path = tmp_path / "pairs.parquet"
    triples_path = tmp_path / "triples.parquet"
    labels_path = tmp_path / "labels.json"
    pairs.to_parquet(pairs_path, index=False)
    triples.to_parquet(triples_path, index=False)
    labels_path.write_text(json.dumps(labels))
    manifest = {
        "benchmark_id": "fixture",
        "seed": 7,
        "scenario": "warm_pair",
        "pairs_path": pairs_path.name,
        "triples_path": triples_path.name,
        "labels_path": labels_path.name,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, manifest


def test_examples_are_loaded_from_fixture_artifacts(manifest_fixture):
    manifest_path, manifest = manifest_fixture

    train = ManifestPolypharmacyDataset.from_manifest(manifest_path, manifest, split="train")
    validation = ManifestPolypharmacyDataset.from_manifest(
        manifest_path, manifest, split="validation"
    )
    test = ManifestPolypharmacyDataset.from_manifest(manifest_path, manifest, split="test")

    assert len(train) == 2
    assert len(validation) == 1
    assert len(test) == 1
    _, _, _, _, train_positive_labels, train_positive = train[0]
    _, _, _, _, train_unlabeled_labels, train_unlabeled = train[1]
    assert train_positive is True
    assert train_positive_labels.tolist() == [1.0, 0.0]
    assert train_unlabeled is False
    assert train_unlabeled_labels.tolist() == [0.0, 0.0]


def test_prevalence_model_matches_fixture_train_labels(manifest_fixture):
    manifest_path, manifest = manifest_fixture
    train = ManifestPolypharmacyDataset.from_manifest(manifest_path, manifest, split="train")
    train_labels = torch.tensor([record["labels"] for record in train.records])
    model = PrevalenceModel(num_labels=train_labels.shape[1])
    model.fit(train_labels)
    expected = train_labels.mean(dim=0)
    assert torch.allclose(model.predict_proba(batch_size=1)[0], expected)


def test_repeated_loading_is_identical_and_features_are_order_stable(manifest_fixture):
    manifest_path, manifest = manifest_fixture

    first = ManifestPolypharmacyDataset.from_manifest(manifest_path, manifest, split="train")
    second = ManifestPolypharmacyDataset.from_manifest(manifest_path, manifest, split="train")
    for first_example, second_example in zip(first, second):
        for first_value, second_value in zip(first_example, second_example):
            if isinstance(first_value, torch.Tensor):
                assert torch.equal(first_value, second_value)
            else:
                assert first_value == second_value


@pytest.mark.parametrize(
    "mutation, expected_message",
    [
        ("duplicate_pair", "pair_id values must be unique"),
        ("missing_column", "missing required columns"),
        ("unknown_pair_reference", "references unknown pair_id"),
        ("unknown_label", "label_cui values must exist in labels.json"),
        ("scenario_mismatch", "scenario values must match manifest scenario"),
    ],
)
def test_malformed_artifacts_fail_clearly(manifest_fixture, mutation, expected_message):
    manifest_path, manifest = manifest_fixture
    pairs_path = manifest_path.parent / manifest["pairs_path"]
    triples_path = manifest_path.parent / manifest["triples_path"]
    pairs = pd.read_parquet(pairs_path)
    triples = pd.read_parquet(triples_path)

    if mutation == "duplicate_pair":
        pairs = pd.concat([pairs, pairs.iloc[[0]]], ignore_index=True)
    elif mutation == "missing_column":
        pairs = pairs.drop(columns=["split"])
    elif mutation == "unknown_pair_reference":
        triples.loc[0, "pair_id"] = "missing_pair"
    elif mutation == "unknown_label":
        triples.loc[0, "label_cui"] = "C999"
    elif mutation == "scenario_mismatch":
        pairs.loc[0, "scenario"] = "cold_1"

    pairs.to_parquet(pairs_path, index=False)
    triples.to_parquet(triples_path, index=False)

    with pytest.raises(ValueError, match=expected_message):
        ManifestPolypharmacyDataset.from_manifest(manifest_path, manifest, split="train")


def test_test_split_can_be_loaded_only_when_requested(manifest_fixture):
    manifest_path, manifest = manifest_fixture

    train = ManifestPolypharmacyDataset.from_manifest(manifest_path, manifest, split="train")
    validation = ManifestPolypharmacyDataset.from_manifest(
        manifest_path, manifest, split="validation"
    )
    assert {record["split"] for record in train.records} == {"train"}
    assert {record["split"] for record in validation.records} == {"validation"}

    test = ManifestPolypharmacyDataset.from_manifest(manifest_path, manifest, split="test")
    assert {record["split"] for record in test.records} == {"test"}

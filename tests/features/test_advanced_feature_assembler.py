import numpy as np
import pytest

from src.features.advanced_feature_assembler import assemble_multimodal_features
from src.features.token_feature_artifact import TokenFeatureArtifact


HASH_A = "a" * 64


def _tokens(tmp_path, name, ids, dim):
    return TokenFeatureArtifact.write(
        tmp_path / f"{name}.npz",
        ids,
        np.ones((len(ids), 2, dim), dtype=np.float32),
        np.zeros((len(ids), 2), dtype=bool),
        np.ones(len(ids), dtype=bool),
        producer=name,
        producer_config={"dim": dim},
        source_sha256=HASH_A,
    )


def test_assembler_joins_all_modalities_in_manifest_drug_order(tmp_path):
    molformer = _tokens(tmp_path, "molformer", ["D2", "D1"], 3)
    mpnn = _tokens(tmp_path, "mpnn", ["D1", "D2"], 4)
    kg = _tokens(tmp_path, "primekg_hgt", ["D2", "D1"], 5)
    artifact = assemble_multimodal_features(
        tmp_path / "advanced.npz",
        required_drug_ids=["D2", "D1"],
        morgan_drug_ids=["D1", "D2"],
        morgan=np.arange(8, dtype=np.float32).reshape(2, 4),
        molformer=molformer,
        mpnn=mpnn,
        kg=kg,
        morgan_provenance_hash=HASH_A,
        manifest_compatibility={
            "benchmark_id": "fixture",
            "scenario": "cold_2",
            "seed": 42,
            "manifest_hash": "manifest42",
        },
    )
    assert artifact.drug_ids == ("D1", "D2")
    assert artifact.molformer_tokens.shape == (2, 2, 3)
    assert artifact.mpnn_tokens.shape == (2, 2, 4)
    assert artifact.kg_tokens.shape == (2, 2, 5)


def test_assembler_fails_closed_on_any_missing_drug(tmp_path):
    molformer = _tokens(tmp_path, "molformer", ["D1"], 3)
    mpnn = _tokens(tmp_path, "mpnn", ["D1", "D2"], 4)
    kg = _tokens(tmp_path, "primekg_hgt", ["D1", "D2"], 5)
    with pytest.raises(ValueError, match="molformer.*D2"):
        assemble_multimodal_features(
            tmp_path / "advanced.npz",
            required_drug_ids=["D1", "D2"],
            morgan_drug_ids=["D1", "D2"],
            morgan=np.ones((2, 4), dtype=np.float32),
            molformer=molformer,
            mpnn=mpnn,
            kg=kg,
            morgan_provenance_hash=HASH_A,
            manifest_compatibility={
                "benchmark_id": "fixture",
                "scenario": "cold_2",
                "seed": 42,
                "manifest_hash": "manifest42",
            },
        )


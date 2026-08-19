import numpy as np
import pytest
import torch

from src.features.cached_token_artifact import build_cached_token_artifact, CachedTokenArtifact
from src.models.multimodal_teacher_student import MultiModalTeacher


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _teacher():
    return MultiModalTeacher(4, 3, 2, 2, 1, 8, 1, 2, num_heads=2, cache_token_count=3, cache_token_dim=6).eval()


def _batch(ids):
    size = len(ids)
    return {
        "drug_ids": ids,
        "inputs": {
            "morgan": torch.randn(size, 4),
            "molecular_tokens": torch.randn(size, 2, 3),
            "kg_tokens": torch.randn(size, 1, 2),
            "morgan_available": torch.ones(size, dtype=torch.bool),
            "molecular_available": torch.ones(size, dtype=torch.bool),
            "kg_available": torch.ones(size, dtype=torch.bool),
        },
    }


def test_build_cached_token_artifact_uses_teacher_cache_encoder(tmp_path):
    path = tmp_path / "cache.npz"
    build_cached_token_artifact(
        _teacher(),
        [_batch(["drug-b", "drug-a"])],
        path,
        teacher_checkpoint_hash=HASH_A,
        teacher_config_hash=HASH_B,
        multimodal_feature_artifact_hash=HASH_C,
    )
    artifact = CachedTokenArtifact.load(path)
    assert artifact.drug_ids == ("drug-a", "drug-b")
    assert artifact.tokens.shape == (2, 3, 6)
    assert artifact.metadata["teacher_checkpoint_hash"] == HASH_A
    assert artifact.metadata["teacher_config_hash"] == HASH_B


@pytest.mark.parametrize(
    "batches, pattern",
    [
        ([_batch(["drug-a"]), _batch(["drug-a"])], "duplicate"),
        ([{"drug_ids": ["drug-a", ""], "inputs": _batch(["x", "y"])["inputs"]}], "drug id"),
    ],
)
def test_build_cached_token_artifact_rejects_duplicate_or_missing_ids(tmp_path, batches, pattern):
    with pytest.raises(ValueError, match=pattern):
        build_cached_token_artifact(
            _teacher(),
            batches,
            tmp_path / "cache.npz",
            teacher_checkpoint_hash=HASH_A,
            teacher_config_hash=HASH_B,
            multimodal_feature_artifact_hash=HASH_C,
        )

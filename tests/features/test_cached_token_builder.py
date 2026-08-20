import hashlib
import json

import numpy as np
import pytest
import torch

from build_teacher_cache import _resolve_teacher_selection
from src.features.cached_token_artifact import build_cached_token_artifact, CachedTokenArtifact
from src.models.multimodal_teacher_student import MultiModalTeacher


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
SELECTION_HASH = "d" * 64


def _teacher():
    return MultiModalTeacher(
        morgan_dim=4,
        molformer_dim=3,
        mpnn_dim=5,
        kg_dim=2,
        molformer_token_count=2,
        mpnn_token_count=3,
        kg_token_count=1,
        hidden_dim=8,
        num_organ=1,
        num_specific=2,
        num_heads=2,
        cache_token_count=3,
        cache_token_dim=6,
    ).eval()


def _batch(ids):
    size = len(ids)
    return {
        "drug_ids": ids,
        "inputs": {
            "morgan": torch.randn(size, 4),
            "molformer_tokens": torch.randn(size, 2, 3),
            "mpnn_tokens": torch.randn(size, 3, 5),
            "kg_tokens": torch.randn(size, 1, 2),
            "morgan_available": torch.ones(size, dtype=torch.bool),
            "molformer_available": torch.ones(size, dtype=torch.bool),
            "mpnn_available": torch.ones(size, dtype=torch.bool),
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
        teacher_selection_hash=SELECTION_HASH,
        teacher_selected_mode="baseline",
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
            teacher_selection_hash=SELECTION_HASH,
            teacher_selected_mode="baseline",
            multimodal_feature_artifact_hash=HASH_C,
        )


def test_cache_selection_hash_is_computed_when_cli_hash_is_omitted(tmp_path):
    path = tmp_path / "teacher_validation_selection.json"
    path.write_text(json.dumps({"selected": {"mode": "baseline", "macro_ap": 0.5}}))

    selection, actual_hash = _resolve_teacher_selection(path)

    assert selection["selected"]["mode"] == "baseline"
    assert actual_hash == hashlib.sha256(path.read_bytes()).hexdigest()


def test_cache_selection_hash_mismatch_is_rejected(tmp_path):
    path = tmp_path / "teacher_validation_selection.json"
    path.write_text(json.dumps({"selected": {"mode": "fused", "macro_ap": 0.5}}))

    with pytest.raises(ValueError, match="hash mismatch"):
        _resolve_teacher_selection(path, expected_hash="0" * 64)

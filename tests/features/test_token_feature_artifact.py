import hashlib

import numpy as np
import pytest

from src.features.token_feature_artifact import TokenFeatureArtifact


HASH_A = "a" * 64


def test_token_artifact_round_trip_is_sorted_safe_and_provenanced(tmp_path):
    path = tmp_path / "tokens.npz"
    artifact = TokenFeatureArtifact.write(
        path,
        ["D2", "D1"],
        np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        np.array([[False, False, True], [False, True, True]], dtype=bool),
        np.array([True, True], dtype=bool),
        producer="molformer",
        producer_config={"model_id": "ibm/test", "revision": "1" * 40},
        source_sha256=HASH_A,
    )

    assert artifact.drug_ids == ("D1", "D2")
    assert artifact.lookup("D2")["tokens"].shape == (3, 4)
    assert artifact.metadata["producer"] == "molformer"
    assert len(artifact.provenance_sha256) == 64
    assert artifact.provenance_sha256 == artifact.metadata["provenance_sha256"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        path.with_name("tokens.npz.sha256").read_text().strip()
    )
    with pytest.raises(ValueError):
        artifact.tokens[0, 0, 0] = 3.0


def test_token_artifact_rejects_bad_masks_duplicates_and_tampering(tmp_path):
    common = dict(
        tokens=np.ones((2, 3, 4), dtype=np.float32),
        padding_mask=np.zeros((2, 3), dtype=bool),
        available=np.ones(2, dtype=bool),
        producer="mpnn",
        producer_config={"hidden_dim": 4},
        source_sha256=HASH_A,
    )
    with pytest.raises(ValueError, match="duplicate"):
        TokenFeatureArtifact.write(tmp_path / "duplicate.npz", ["D1", "D1"], **common)
    with pytest.raises(ValueError, match="padding_mask"):
        TokenFeatureArtifact.write(
            tmp_path / "mask.npz",
            ["D1", "D2"],
            **(common | {"padding_mask": np.zeros((2, 2), dtype=bool)}),
        )
    assert not (tmp_path / "mask.npz").exists()
    valid = tmp_path / "valid.npz"
    TokenFeatureArtifact.write(valid, ["D1", "D2"], **common)
    valid.write_bytes(valid.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        TokenFeatureArtifact.load(valid)

import numpy as np
import pytest

from src.features.cached_token_artifact import CachedTokenArtifact

TEACHER_HASH = "a" * 64
MODALITY_HASH = "b" * 64
SELECTION_HASH = "c" * 64


def _arrays():
    ids = ["drug-b", "drug-a"]
    tokens = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    available = np.ones((2, 3), dtype=bool)
    return ids, tokens, available


def test_cached_tokens_are_sorted_and_round_trip_without_pickle(tmp_path):
    path = tmp_path / "tokens.npz"
    ids, tokens, available = _arrays()
    CachedTokenArtifact.write(
        path,
        ids,
        tokens,
        available,
        teacher_provenance_hash=TEACHER_HASH,
        modality_provenance_hash=MODALITY_HASH,
        teacher_selection_hash=SELECTION_HASH,
        teacher_selected_mode="baseline",
    )
    artifact = CachedTokenArtifact.load(path)
    assert artifact._id_to_index["drug-a"] == 0
    assert artifact._id_to_index["drug-b"] == 1
    assert artifact.drug_ids == ("drug-a", "drug-b")
    assert artifact.lookup("drug-a")[0].shape == (3, 4)
    assert np.array_equal(artifact.lookup_pair("drug-b", "drug-a")[0], tokens[0])


@pytest.mark.parametrize(
    "ids, tokens, message",
    [
        (["drug-a", "drug-a"], np.zeros((2, 3, 4), dtype=np.float32), "duplicate"),
        (["drug-a", ""], np.zeros((2, 3, 4), dtype=np.float32), "drug id"),
        (["drug-a"], np.zeros((1, 3, 4), dtype=np.float64), "float32"),
        (["drug-a"], np.full((1, 3, 4), np.nan, dtype=np.float32), "finite"),
    ],
)
def test_cached_tokens_reject_invalid_inputs(tmp_path, ids, tokens, message):
    with pytest.raises(ValueError, match=message):
        CachedTokenArtifact.write(
            tmp_path / "invalid.npz",
            ids,
            tokens,
            np.ones((len(ids), 3), dtype=bool),
            teacher_provenance_hash=TEACHER_HASH,
            modality_provenance_hash=MODALITY_HASH,
            teacher_selection_hash=SELECTION_HASH,
            teacher_selected_mode="baseline",
        )


def test_cached_tokens_reject_tampering(tmp_path):
    path = tmp_path / "tokens.npz"
    ids, tokens, available = _arrays()
    CachedTokenArtifact.write(
        path,
        ids,
        tokens,
        available,
        teacher_provenance_hash=TEACHER_HASH,
        modality_provenance_hash=MODALITY_HASH,
        teacher_selection_hash=SELECTION_HASH,
        teacher_selected_mode="baseline",
    )
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 0x01
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="checksum|artifact"):
        CachedTokenArtifact.load(path)


def test_cached_tokens_reject_unknown_drug(tmp_path):
    path = tmp_path / "tokens.npz"
    ids, tokens, available = _arrays()
    CachedTokenArtifact.write(
        path,
        ids,
        tokens,
        available,
        teacher_provenance_hash=TEACHER_HASH,
        modality_provenance_hash=MODALITY_HASH,
        teacher_selection_hash=SELECTION_HASH,
        teacher_selected_mode="baseline",
    )
    artifact = CachedTokenArtifact.load(path)
    with pytest.raises(KeyError, match="unknown"):
        artifact.lookup("unknown")


def test_cached_tokens_require_canonical_hashes(tmp_path):
    ids, tokens, available = _arrays()
    with pytest.raises(ValueError, match="SHA-256|hash"):
        CachedTokenArtifact.write(
            tmp_path / "invalid.npz",
            ids,
            tokens,
            available,
            teacher_provenance_hash="teacher-hash",
            modality_provenance_hash=MODALITY_HASH,
            teacher_selection_hash=SELECTION_HASH,
            teacher_selected_mode="baseline",
        )


def test_cached_tokens_are_immutable_but_lookups_are_owned_copies(tmp_path):
    path = tmp_path / "tokens.npz"
    ids, tokens, available = _arrays()
    CachedTokenArtifact.write(
        path,
        ids,
        tokens,
        available,
        teacher_provenance_hash=TEACHER_HASH,
        modality_provenance_hash=MODALITY_HASH,
        teacher_selection_hash=SELECTION_HASH,
        teacher_selected_mode="baseline",
    )
    artifact = CachedTokenArtifact.load(path)
    with pytest.raises(ValueError):
        artifact.tokens[0, 0, 0] = 999.0
    with pytest.raises(TypeError):
        artifact.metadata["new"] = "value"
    looked_up, _ = artifact.lookup("drug-a")
    looked_up[0, 0] = 999.0
    assert artifact.lookup("drug-a")[0][0, 0] != 999.0


def test_cached_tokens_record_teacher_selection_provenance(tmp_path):
    path = tmp_path / "tokens.npz"
    ids, tokens, available = _arrays()
    artifact = CachedTokenArtifact.write(
        path,
        ids,
        tokens,
        available,
        teacher_provenance_hash=TEACHER_HASH,
        modality_provenance_hash=MODALITY_HASH,
        teacher_selection_hash=SELECTION_HASH,
        teacher_selected_mode="fused",
    )
    assert artifact.metadata["teacher_selection_hash"] == SELECTION_HASH
    assert artifact.metadata["teacher_selected_mode"] == "fused"


def test_cached_tokens_reject_invalid_teacher_selection_mode(tmp_path):
    ids, tokens, available = _arrays()
    with pytest.raises(ValueError, match="teacher_selected_mode"):
        CachedTokenArtifact.write(
            tmp_path / "invalid-selection.npz",
            ids,
            tokens,
            available,
            teacher_provenance_hash=TEACHER_HASH,
            modality_provenance_hash=MODALITY_HASH,
            teacher_selection_hash=SELECTION_HASH,
            teacher_selected_mode="invalid",
        )

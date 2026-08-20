import pytest
import torch

from scripts.train_primekg_hgt import _bounded_supervision


def test_bounded_supervision_is_deterministic_and_preserves_edge_pairs():
    positives = torch.arange(80, dtype=torch.long).reshape(2, 40)

    first = _bounded_supervision(positives, 7, seed=42)
    second = _bounded_supervision(positives, 7, seed=42)
    different = _bounded_supervision(positives, 7, seed=43)

    assert first.shape == (2, 7)
    assert torch.equal(first, second)
    assert not torch.equal(first, different)
    assert all(
        any(torch.equal(first[:, index], positives[:, source]) for source in range(40))
        for index in range(first.shape[1])
    )


def test_bounded_supervision_keeps_small_relations_and_rejects_bad_limits():
    positives = torch.arange(12, dtype=torch.long).reshape(2, 6)

    assert _bounded_supervision(positives, None, seed=1) is positives
    assert _bounded_supervision(positives, 6, seed=1) is positives
    with pytest.raises(ValueError, match="positive integer"):
        _bounded_supervision(positives, 0, seed=1)

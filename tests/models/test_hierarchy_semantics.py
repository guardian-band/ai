import json

import pytest
import torch

from src.models.hierarchy import HierarchyLoss, load_hierarchy_mapping
from src.models.token_cross_attention import TokenCrossAttention


def test_hierarchy_loss_uses_separate_specific_and_organ_heads():
    loss_fn = HierarchyLoss({0: 1, 1: 0})
    specific = torch.tensor([[0.0, 2.0]])
    organ = torch.tensor([[1.0, -1.0]])
    expected = torch.relu(torch.sigmoid(specific[:, [0, 1]]) - torch.sigmoid(organ[:, [1, 0]])).mean()
    assert torch.allclose(loss_fn(specific, organ), expected)


def test_hierarchy_loss_rejects_missing_mapping_and_bad_shapes():
    with pytest.raises(ValueError, match="complete mapping"):
        HierarchyLoss({0: 0}, num_specific=2)
    loss_fn = HierarchyLoss({0: 0})
    with pytest.raises(ValueError, match="2D"):
        loss_fn(torch.zeros(2), torch.zeros(2, 1))
    with pytest.raises(ValueError, match="out of bounds"):
        HierarchyLoss({0: 2})(torch.zeros(1, 1), torch.zeros(1, 2))


def test_hierarchy_mapping_loader_is_versioned_and_complete(tmp_path):
    path = tmp_path / "mapping.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mappings": [
                    {"specific_cui": "C002", "organ_cui": "O2"},
                    {"specific_cui": "C001", "organ_cui": "O1"},
                ],
            }
        )
    )
    mapping = load_hierarchy_mapping(path, selected_specific_cuis=["C001", "C002"])
    assert mapping.organ_order == ("O1", "O2")
    assert mapping.specific_to_organ == {"C001": 0, "C002": 1}

    with pytest.raises(ValueError, match="complete"):
        load_hierarchy_mapping(path, selected_specific_cuis=["C001", "C003"])


def test_token_cross_attention_handles_all_padding_without_nan():
    attention = TokenCrossAttention(embed_dim=4, num_heads=2, dropout=0.0).eval()
    tokens_a = torch.randn(1, 3, 4)
    tokens_b = torch.randn(1, 3, 4)
    masks = torch.ones(1, 3, dtype=torch.bool)
    pooled_a, pooled_b = attention(tokens_a, tokens_b, masks, masks)
    assert torch.isfinite(pooled_a).all()
    assert torch.isfinite(pooled_b).all()

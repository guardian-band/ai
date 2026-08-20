import pandas as pd
import pytest
import torch

from src.data.primekg_typed_graph import (
    LEAKAGE_RELATIONS,
    build_leakage_safe_edge_partitions,
    prepare_typed_primekg,
    typed_edge_index_sha256,
    validate_typed_edge_index_sha256,
)


def test_typed_graph_preserves_node_and_relation_types_and_removes_leakage():
    frame = pd.DataFrame(
        [
            {"x_id": "D1", "x_type": "drug", "relation": "drug_protein", "y_id": "G1", "y_type": "gene/protein"},
            {"x_id": "D2", "x_type": "drug", "relation": "drug_protein", "y_id": "G2", "y_type": "gene/protein"},
            {"x_id": "G1", "x_type": "gene/protein", "relation": "protein_disease", "y_id": "X1", "y_type": "disease"},
            {"x_id": "D1", "x_type": "drug", "relation": "drug_effect", "y_id": "E1", "y_type": "effect/phenotype"},
        ]
    )
    prepared = prepare_typed_primekg(frame)
    assert "gene_protein" in prepared.node_ids
    assert "disease" in prepared.node_ids
    assert all(edge_type[1] not in LEAKAGE_RELATIONS for edge_type in prepared.edge_index)
    assert ("drug", "drug_protein", "gene_protein") in prepared.edge_index
    assert ("gene_protein", "rev_drug_protein", "drug") in prepared.edge_index
    assert prepared.rejected_leakage_edges == 1


def test_typed_graph_rejects_missing_schema_and_duplicate_node_features():
    with pytest.raises(ValueError, match="missing required"):
        prepare_typed_primekg(pd.DataFrame({"x_id": ["D1"]}))


def test_supervision_edge_and_its_reverse_are_absent_from_message_graph():
    rows = [
        {
            "x_id": f"D{index % 4}",
            "x_type": "drug",
            "relation": "drug_protein",
            "y_id": f"G{index}",
            "y_type": "gene/protein",
        }
        for index in range(10)
    ]
    prepared = prepare_typed_primekg(pd.DataFrame(rows))
    message, train, validation = build_leakage_safe_edge_partitions(
        prepared.edge_index, seed=3, train_ratio=0.2, validation_ratio=0.2
    )
    forward = ("drug", "drug_protein", "gene_protein")
    reverse = ("gene_protein", "rev_drug_protein", "drug")
    held_out = torch.cat([train[forward], validation[forward]], dim=1)
    message_forward = set(map(tuple, message[forward].t().tolist()))
    message_reverse = set(map(tuple, message[reverse].t().tolist()))
    for source, destination in held_out.t().tolist():
        assert (source, destination) not in message_forward
        assert (destination, source) not in message_reverse


def test_message_graph_hash_is_order_independent_but_rejects_edge_perturbations():
    edge_type = ("drug", "binds", "gene")
    first = {edge_type: torch.tensor([[1, 0], [0, 1]], dtype=torch.long)}
    reordered = {edge_type: torch.tensor([[0, 1], [1, 0]], dtype=torch.long)}
    perturbed = {edge_type: torch.tensor([[1, 0], [0, 0]], dtype=torch.long)}
    expected = typed_edge_index_sha256(first)

    assert typed_edge_index_sha256(reordered) == expected
    validate_typed_edge_index_sha256(first, expected)
    with pytest.raises(ValueError, match="graph hash"):
        validate_typed_edge_index_sha256(perturbed, expected)

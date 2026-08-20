import torch
import pytest

from src.models.primekg_hgt import (
    PrimeKGHGT,
    TypedLinkPredictor,
    export_hgt_token_artifact,
    load_hgt_checkpoint,
    sample_bipartite_negatives,
    save_hgt_checkpoint,
    split_relation_edges,
)


def test_typed_link_predictor_scales_bilinear_score_and_learns_relation_bias():
    edge_type = ("drug", "targets", "gene")
    predictor = TypedLinkPredictor((edge_type,), hidden_dim=4)
    x_dict = {
        "drug": torch.nn.functional.normalize(torch.ones(1, 4), dim=-1),
        "gene": torch.nn.functional.normalize(torch.ones(1, 4), dim=-1),
    }
    edge_label_index = torch.tensor([[0], [0]], dtype=torch.long)

    score = predictor(x_dict, edge_type, edge_label_index)

    assert score.shape == (1,)
    assert score.item() == pytest.approx(0.5)
    assert torch.isfinite(score).all()
    assert predictor.relation_bias
    relation_key = next(iter(predictor.relation_bias))
    assert predictor.relation_bias[relation_key].shape == torch.Size([])


def test_relation_split_is_deterministic_disjoint_and_keeps_message_edges_separate():
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 1, 0, 2, 1, 3, 2, 4, 3, 4]],
        dtype=torch.long,
    )
    first = split_relation_edges(edge_index, seed=42, train_ratio=0.2, validation_ratio=0.2)
    second = split_relation_edges(edge_index, seed=42, train_ratio=0.2, validation_ratio=0.2)
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    encoded = [set((part[0] * 10 + part[1]).tolist()) for part in first]
    assert not (encoded[0] & encoded[1])
    assert not (encoded[0] & encoded[2])
    assert not (encoded[1] & encoded[2])
    assert sum(part.shape[1] for part in first) == edge_index.shape[1]


def test_negative_sampling_never_returns_a_known_positive():
    positives = torch.tensor([[0, 0, 1, 2], [0, 1, 1, 2]], dtype=torch.long)
    negatives = sample_bipartite_negatives(
        positives,
        num_source_nodes=3,
        num_destination_nodes=4,
        count=5,
        seed=7,
    )
    positive_pairs = set(zip(positives[0].tolist(), positives[1].tolist()))
    negative_pairs = list(zip(negatives[0].tolist(), negatives[1].tolist()))
    assert len(set(negative_pairs)) == 5
    assert not positive_pairs.intersection(negative_pairs)


def test_typed_hgt_emits_layerwise_drug_tokens_and_gradients():
    metadata = (
        ("drug", "gene"),
        (
            ("drug", "targets", "gene"),
            ("gene", "rev_targets", "drug"),
        ),
    )
    model = PrimeKGHGT(
        metadata,
        input_dims={"drug": 4, "gene": 3},
        hidden_dim=8,
        layers=2,
        heads=2,
    )
    x_dict = {"drug": torch.randn(3, 4), "gene": torch.randn(2, 3)}
    edge_index_dict = {
        ("drug", "targets", "gene"): torch.tensor([[0, 1, 2], [0, 1, 0]]),
        ("gene", "rev_targets", "drug"): torch.tensor([[0, 1, 0], [0, 1, 2]]),
    }
    output, tokens = model(x_dict, edge_index_dict, return_drug_tokens=True)
    assert output["drug"].shape == (3, 8)
    assert tokens.shape == (3, 3, 8)
    loss = output["drug"].square().mean() + output["gene"].square().mean()
    loss.backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)


def test_hgt_checkpoint_round_trip_and_token_export(tmp_path):
    metadata = (
        ("drug", "gene"),
        (
            ("drug", "targets", "gene"),
            ("gene", "rev_targets", "drug"),
        ),
    )
    model = PrimeKGHGT(
        metadata,
        input_dims={"drug": 4, "gene": 3},
        hidden_dim=8,
        layers=2,
        heads=2,
        dropout=0.0,
    ).eval()
    checkpoint = tmp_path / "hgt.pt"
    checksum = save_hgt_checkpoint(
        model,
        checkpoint,
        source_sha256="a" * 64,
        training_config={"scenario": "cold_1", "seed": 42},
    )
    loaded, metadata_dict = load_hgt_checkpoint(checkpoint, expected_sha256=checksum)
    x_dict = {"drug": torch.randn(3, 4), "gene": torch.randn(2, 3)}
    edge_index_dict = {
        ("drug", "targets", "gene"): torch.tensor([[0, 1, 2], [0, 1, 0]]),
        ("gene", "rev_targets", "drug"): torch.tensor([[0, 1, 0], [0, 1, 2]]),
    }
    artifact = export_hgt_token_artifact(
        loaded,
        drug_ids=["D1", "D2", "D3"],
        x_dict=x_dict,
        edge_index_dict=edge_index_dict,
        output_path=tmp_path / "hgt_tokens.npz",
        source_sha256="a" * 64,
        checkpoint_sha256=checksum,
    )
    assert metadata_dict["model_config"]["hidden_dim"] == 8
    assert artifact.tokens.shape == (3, 3, 8)
    assert artifact.metadata["producer"] == "primekg_hgt"

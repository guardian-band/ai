import hashlib
import math

import pytest
import torch

from scripts.train_primekg_hgt import (
    _bounded_supervision,
    _filter_incident_drugs,
    _hgt_graph_provenance,
    _fixed_validation_negatives,
    _link_prediction_metrics,
    _require_export_ready,
    _validate_hgt_message_graph_artifact,
    _training_metrics_paths,
    _write_training_metrics,
    select_hgt_checkpoint,
)
from src.data.primekg_typed_graph import PreparedTypedPrimeKG, build_leakage_safe_edge_partitions
from src.features.token_feature_artifact import TokenFeatureArtifact
from src.models.primekg_hgt import PrimeKGHGT


def _prepared_graph(edge_index=None):
    relation = ("drug", "binds", "gene")
    reverse = ("gene", "rev_binds", "drug")
    if edge_index is None:
        edge_index = torch.tensor([[0, 1, 2], [0, 1, 0]], dtype=torch.long)
    return PreparedTypedPrimeKG(
        node_ids={"drug": ("D0", "D1", "D2"), "gene": ("G0", "G1")},
        edge_index={relation: edge_index, reverse: edge_index.flip(0).contiguous()},
        rejected_leakage_edges=0,
        duplicate_edges=0,
    )


def test_cold_message_graph_isolates_excluded_drugs_and_ignores_incident_edge_perturbations():
    prepared = _prepared_graph()
    excluded = {"D1"}
    first = _filter_incident_drugs(prepared, excluded)
    perturbed_edges = torch.cat(
        [prepared.edge_index[("drug", "binds", "gene")], torch.tensor([[1], [1]])], dim=1
    )
    perturbed = _prepared_graph(perturbed_edges)
    second = _filter_incident_drugs(perturbed, excluded)

    assert all(
        torch.equal(first[edge_type], second[edge_type])
        for edge_type in first
    )
    for edge_type, edges in first.items():
        source_type, _, destination_type = edge_type
        if source_type == "drug":
            assert 1 not in edges[0].tolist()
        if destination_type == "drug":
            assert 1 not in edges[1].tolist()
    model = PrimeKGHGT(
        prepared.metadata,
        input_dims={"drug": 3, "gene": 3},
        hidden_dim=4,
        layers=1,
        heads=1,
        dropout=0.0,
    ).eval()
    features = {
        "drug": torch.eye(3, dtype=torch.float32),
        "gene": torch.eye(2, 3, dtype=torch.float32),
    }
    with torch.inference_mode():
        _, first_tokens = model(features, first, return_drug_tokens=True)
        _, second_tokens = model(features, second, return_drug_tokens=True)
    assert torch.equal(first_tokens[1], second_tokens[1])


def test_warm_message_graph_hash_excludes_train_and_validation_supervision_edges():
    prepared = _prepared_graph(
        torch.tensor(
            [[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]], dtype=torch.long
        )
    )
    message, train, validation = build_leakage_safe_edge_partitions(
        prepared.edge_index, seed=11, train_ratio=0.2, validation_ratio=0.2
    )
    provenance = _hgt_graph_provenance(
        message,
        train,
        validation,
        excluded_drug_ids=set(),
        required_drug_ids=("D0", "D1", "D2"),
        drug_ids=prepared.node_ids["drug"],
    )

    assert provenance["cold_protocol"] == "train_message_graph_only"
    for edge_type, held_out in {**train, **validation}.items():
        if edge_type not in message:
            continue
        assert not set(map(tuple, held_out.t().tolist())).intersection(
            set(map(tuple, message[edge_type].t().tolist()))
        )
    assert provenance["relation_edge_counts"]["drug\x1fbinds\x1fgene"]["validation"] == validation[("drug", "binds", "gene")].shape[1]


def test_hgt_artifact_message_graph_hash_is_validated(tmp_path):
    edge_type = ("drug", "binds", "gene")
    edges = {edge_type: torch.tensor([[0, 1], [0, 1]], dtype=torch.long)}
    provenance = _hgt_graph_provenance(
        edges,
        {},
        {},
        excluded_drug_ids=set(),
        required_drug_ids=("D0", "D1"),
        drug_ids=("D0", "D1"),
    )
    artifact = TokenFeatureArtifact.write(
        tmp_path / "kg.npz",
        ["D0", "D1"],
        torch.zeros(2, 1, 2).numpy(),
        torch.zeros(2, 1, dtype=torch.bool).numpy(),
        torch.ones(2, dtype=torch.bool).numpy(),
        producer="primekg_hgt",
        producer_config=provenance,
        source_sha256="a" * 64,
    )

    _validate_hgt_message_graph_artifact(artifact, edges)
    altered = {edge_type: torch.tensor([[0, 0], [0, 1]], dtype=torch.long)}
    with pytest.raises(ValueError, match="graph hash"):
        _validate_hgt_message_graph_artifact(artifact, altered)


def test_link_prediction_metrics_report_per_relation_fields_and_global_ap():
    relation_outputs = {
        ("drug", "targets", "gene"): (
            torch.tensor([4.0, 3.0, -2.0, -3.0]),
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
        ),
        ("drug", "binds", "protein"): (
            torch.tensor([0.0, 0.5, 1.0]),
            torch.tensor([1.0, 0.0, 0.0]),
        ),
    }

    metrics = _link_prediction_metrics(relation_outputs)

    assert metrics["global"]["ap"] > 0.8
    assert metrics["global"]["auroc"] is not None
    assert metrics["global"]["positive_count"] == 3
    assert metrics["global"]["negative_count"] == 4
    assert metrics["macro_relation_ap"] == pytest.approx(
        sum(item["ap"] for item in metrics["relations"].values()) / 2
    )
    for relation in metrics["relations"].values():
        assert set(
            (
                "bce",
                "ap",
                "auroc",
                "positive_logit_mean",
                "positive_logit_std",
                "negative_logit_mean",
                "negative_logit_std",
                "positive_count",
                "negative_count",
            )
        ).issubset(relation)


def test_link_prediction_metrics_preserve_undefined_auroc_as_null():
    metrics = _link_prediction_metrics(
        {
            ("drug", "targets", "gene"): (
                torch.tensor([0.0, 1.0, 2.0]),
                torch.tensor([1.0, 1.0, 1.0]),
            )
        }
    )

    assert metrics["global"]["auroc"] is None
    assert metrics["relations"][
        "drug\x1ftargets\x1fgene"
    ]["auroc"] is None


def test_link_prediction_metrics_fail_fast_for_nonfinite_inputs():
    with pytest.raises(FloatingPointError, match="non-finite"):
        _link_prediction_metrics(
            {
                ("drug", "targets", "gene"): (
                    torch.tensor([math.inf, 0.0]),
                    torch.tensor([1.0, 0.0]),
                )
            }
        )


def test_separable_relation_beats_negative_sampling_baseline_and_is_exportable():
    metrics = _link_prediction_metrics(
        {
            ("drug", "targets", "gene"): (
                torch.tensor([5.0, 4.0, -4.0, -5.0]),
                torch.tensor([1.0, 1.0, 0.0, 0.0]),
            )
        }
    )

    assert metrics["global"]["ap"] > 0.90
    _require_export_ready(metrics)


def test_export_gate_rejects_uninterpretable_high_validation_bce():
    metrics = _link_prediction_metrics(
        {
            ("drug", "targets", "gene"): (
                torch.tensor([-5.0, -5.0, -6.0, -6.0]),
                torch.tensor([1.0, 1.0, 0.0, 0.0]),
            )
        }
    )

    with pytest.raises(RuntimeError, match="BCE exceeds 2.0"):
        _require_export_ready(metrics)


def test_validation_negative_edges_are_fixed_and_training_seed_can_change_them():
    supervision = {
        ("drug", "targets", "gene"): torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    }
    all_edges = {
        ("drug", "targets", "gene"): torch.tensor(
            [[0, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.long
        )
    }
    node_ids = {"drug": ("D0", "D1", "D2"), "gene": ("G0", "G1", "G2")}

    first = _fixed_validation_negatives(supervision, all_edges, node_ids, seed=7)
    second = _fixed_validation_negatives(supervision, all_edges, node_ids, seed=7)

    assert torch.equal(first[next(iter(first))], second[next(iter(second))])


def test_checkpoint_selection_prefers_macro_ap_then_bce_then_earlier_epoch():
    history = [
        {"epoch": 3, "validation": {"macro_relation_ap": 0.8, "bce": 0.4}},
        {"epoch": 2, "validation": {"macro_relation_ap": 0.8, "bce": 0.4}},
        {"epoch": 1, "validation": {"macro_relation_ap": 0.8, "bce": 0.5}},
        {"epoch": 4, "validation": {"macro_relation_ap": 0.7, "bce": 0.1}},
    ]

    selected = select_hgt_checkpoint(history)

    assert selected["epoch"] == 2


def test_training_metrics_paths_are_checkpoint_specific_and_atomic(tmp_path):
    first_metrics, first_sidecar = _training_metrics_paths(tmp_path / "hgt_seed_42.pt")
    second_metrics, second_sidecar = _training_metrics_paths(tmp_path / "hgt_seed_101.pt")

    assert first_metrics != second_metrics
    assert first_sidecar != second_sidecar
    payload = {"schema_version": 1, "seed": 42}
    checksum = _write_training_metrics(first_metrics, payload)
    data = first_metrics.read_bytes()
    assert checksum == hashlib.sha256(data).hexdigest()
    assert first_sidecar.read_text(encoding="ascii").strip() == checksum
    assert second_metrics.name == "hgt_seed_101.pt.training_metrics.json"
    assert second_sidecar.name == "hgt_seed_101.pt.training_metrics.json.sha256"


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

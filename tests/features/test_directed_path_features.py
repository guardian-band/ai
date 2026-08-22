import numpy as np
import pandas as pd

from src.features.directed_path_features import (
    DirectedPathFeatureIndex,
    degree_matched_permutation,
    label_specific_degree_adjusted_enrichment,
)


def test_directed_features_count_targets_and_ppi_paths():
    graph = pd.DataFrame(
        [
            ("A", "drug", "drug_protein", "P1", "gene/protein"),
            ("B", "drug", "drug_protein", "P1", "gene/protein"),
            ("A", "drug", "drug_protein", "P2", "gene/protein"),
            ("P2", "gene/protein", "protein_protein", "P3", "gene/protein"),
            ("B", "drug", "drug_protein", "P3", "gene/protein"),
            ("A", "drug", "drug_effect", "L", "effect"),
        ],
        columns=["x_id", "x_type", "relation", "y_id", "y_type"],
    )
    index = DirectedPathFeatureIndex(graph, ["L"])
    values = index.pair_features("A", "B")
    assert np.expm1(values[0]) == 1
    assert np.isclose(values[1], 1 / 3)
    assert np.expm1(values[2]) == 1
    assert np.expm1(values[4]) >= 1
    assert values[7] == 1
    assert any("fwd:" in relation for relation in index.first_hop["A"].values())


def test_degree_matched_permutation_is_deterministic():
    features = np.asarray([[1, 1], [2, 1], [1, 2], [2, 2]], dtype=float)
    positive = np.asarray([True, False, True, False])
    first = degree_matched_permutation(features, positive, degree_column=1, samples=20, seed=7)
    second = degree_matched_permutation(features, positive, degree_column=1, samples=20, seed=7)
    assert first == second


def test_label_specific_enrichment_is_fdr_reported_and_deterministic():
    features = np.zeros((80, 8), dtype=np.float32)
    targets = np.zeros((80, 2), dtype=bool)
    targets[:40, 0] = True
    targets[::2, 1] = True
    features[:40, 0] = 3.0
    first = label_specific_degree_adjusted_enrichment(features, targets, ["signal", "noise"])
    second = label_specific_degree_adjusted_enrichment(features, targets, ["signal", "noise"])
    assert first == second
    assert first["fdr_method"] == "Benjamini-Hochberg"
    assert first["significant_positive_labels"] == 1
    assert first["top_enrichments"][0]["label"] == "signal"

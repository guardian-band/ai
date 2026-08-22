import numpy as np
import pandas as pd

from src.features.directed_path_features import DirectedPathFeatureIndex, degree_matched_permutation


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

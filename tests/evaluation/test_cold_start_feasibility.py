import pandas as pd

from src.evaluation.cold_start_feasibility import audit_safe_paths, evaluate_similarity_transfer


def test_path_audit_removes_target_and_ddi_leakage():
    graph = pd.DataFrame(
        [
            ("A", "drug", "drug_protein", "P", "gene_protein"),
            ("P", "gene_protein", "protein_drug", "B", "drug"),
            ("A", "drug", "drug_effect", "L", "effect"),
            ("L", "effect", "related", "B", "drug"),
            ("C", "drug", "drug_drug", "D", "drug"),
        ],
        columns=["x_id", "x_type", "relation", "y_id", "y_type"],
    )
    pairs = pd.DataFrame(
        [("p1", "A", "B", "validation"), ("p2", "C", "D", "test")],
        columns=["pair_id", "drug_a", "drug_b", "split"],
    )
    report = audit_safe_paths(graph, pairs, ["L"])
    assert report["splits"]["validation"]["shortest_path_hops"]["2"] == 1
    assert report["splits"]["test"]["coverage"] == 0
    assert report["filtering"]["relation_edges"] == 2


def test_similarity_transfer_uses_only_train_pair_profiles():
    pairs = pd.DataFrame(
        [
            ("p1", "A", "B", "train"),
            ("p2", "A", "C", "train"),
            ("p3", "X", "B", "validation"),
        ],
        columns=["pair_id", "drug_a", "drug_b", "split"],
    )
    triples = pd.DataFrame(
        [("p1", "A", "B", "L1", "observed_positive"), ("p2", "A", "C", "L2", "observed_positive")],
        columns=["pair_id", "drug_a", "drug_b", "label_cui", "observation_status"],
    )
    morgan = pd.DataFrame(
        [("A", [1, 0]), ("B", [0, 1]), ("C", [1, 1]), ("X", [1, 0])],
        columns=["drugbank_id", "morgan_fingerprint"],
    )
    result = evaluate_similarity_transfer(pairs, triples, ["L1", "L2"], morgan, neighbors=(1,))
    assert result.predictions.shape == (1, 2)
    assert result.predictions[0, 0] == 1.0
    assert result.report["transfer_coverage"] == 1.0

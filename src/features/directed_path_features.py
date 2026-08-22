"""Small directed PrimeKG path features for cold-start signal probing."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from src.data.primekg_typed_graph import LEAKAGE_RELATIONS, REQUIRED_COLUMNS


def _normal(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


class DirectedPathFeatureIndex:
    """Index strict safe edges while retaining forward/inverse relation identity."""

    HASH_BUCKETS = 32
    BASE_NAMES = (
        "shared_targets",
        "target_jaccard",
        "two_hop_paths",
        "three_hop_paths",
        "three_hop_protein_paths",
        "minimum_protein_degree",
        "mean_protein_degree",
        "connected_within_three_hops",
    )

    def __init__(self, frame: pd.DataFrame, target_labels: Iterable[str]):
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"PrimeKG missing columns: {', '.join(sorted(missing))}")
        values = frame[["x_id", "x_type", "relation", "y_id", "y_type"]].dropna().copy()
        for column in ("x_id", "y_id"):
            values[column] = values[column].astype(str).str.strip()
        for column in ("x_type", "relation", "y_type"):
            values[column] = values[column].map(_normal)
        labels = set(map(str, target_labels))
        values = values.loc[
            ~values["relation"].isin(LEAKAGE_RELATIONS)
            & ~values["x_id"].isin(labels)
            & ~values["y_id"].isin(labels)
        ].drop_duplicates(ignore_index=True)
        self.adjacency: dict[tuple[str, str], dict[tuple[str, str], str]] = defaultdict(dict)
        self.drug_nodes: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for row in values.itertuples(index=False):
            left = (str(row.x_type), str(row.x_id))
            right = (str(row.y_type), str(row.y_id))
            relation = str(row.relation)
            self.adjacency[left].setdefault(right, f"fwd:{relation}")
            self.adjacency[right].setdefault(left, f"inv:{relation}")
            if "drug" in left[0]:
                self.drug_nodes[left[1]].add(left)
            if "drug" in right[0]:
                self.drug_nodes[right[1]].add(right)
        self.first_hop = {
            drug: {
                neighbor: relation
                for node in nodes
                for neighbor, relation in self.adjacency.get(node, {}).items()
            }
            for drug, nodes in self.drug_nodes.items()
        }

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.BASE_NAMES + tuple(f"relation_path_bucket_{i:02d}" for i in range(self.HASH_BUCKETS))

    @staticmethod
    def _is_protein(node: tuple[str, str]) -> bool:
        return "protein" in node[0] or "gene" in node[0]

    @classmethod
    def _bucket(cls, signature: str) -> int:
        return int.from_bytes(hashlib.sha256(signature.encode()).digest()[:4], "big") % cls.HASH_BUCKETS

    def pair_features(self, drug_a: str, drug_b: str) -> np.ndarray:
        left = self.first_hop.get(str(drug_a), {})
        right = self.first_hop.get(str(drug_b), {})
        common = left.keys() & right.keys()
        left_targets = {node for node in left if self._is_protein(node)}
        right_targets = {node for node in right if self._is_protein(node)}
        shared_targets = left_targets & right_targets
        target_union = left_targets | right_targets
        buckets = np.zeros(self.HASH_BUCKETS, dtype=np.float64)
        for middle in common:
            signature = f"{left[middle]}|{middle[0]}|{right[middle]}"
            buckets[self._bucket(signature)] += 1.0

        three_hop = 0
        protein_three_hop = 0
        protein_degrees: list[int] = []
        outer, target = (left, right) if len(left) <= len(right) else (right, left)
        target_nodes = target.keys()
        for first, edge_one in outer.items():
            if self._is_protein(first):
                protein_degrees.append(len(self.adjacency.get(first, {})))
            for second in self.adjacency.get(first, {}).keys() & target_nodes:
                three_hop += 1
                edge_two = self.adjacency[first][second]
                edge_three = target[second]
                signature = f"{edge_one}|{first[0]}|{edge_two}|{second[0]}|{edge_three}"
                buckets[self._bucket(signature)] += 1.0
                if self._is_protein(first) and self._is_protein(second):
                    protein_three_hop += 1
                    protein_degrees.append(len(self.adjacency.get(second, {})))
        degree_min = min(protein_degrees) if protein_degrees else 0
        degree_mean = float(np.mean(protein_degrees)) if protein_degrees else 0.0
        base = np.asarray(
            [
                len(shared_targets),
                len(shared_targets) / len(target_union) if target_union else 0.0,
                len(common),
                three_hop,
                protein_three_hop,
                degree_min,
                degree_mean,
                float(bool(common or three_hop)),
            ],
            dtype=np.float64,
        )
        # Counts and degrees are heavy-tailed; log scaling reduces hub dominance.
        base[[0, 2, 3, 4, 5, 6]] = np.log1p(base[[0, 2, 3, 4, 5, 6]])
        return np.concatenate([base, np.log1p(buckets)]).astype(np.float32)

    def transform_records(self, records: Iterable[Mapping[str, Any]]) -> np.ndarray:
        return np.stack(
            [self.pair_features(str(row["drug_a"]), str(row["drug_b"])) for row in records]
        )


def degree_matched_permutation(
    features: np.ndarray,
    positive: np.ndarray,
    *,
    degree_column: int = 6,
    samples: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare positives and controls while permuting within graph-degree strata."""

    positive = np.asarray(positive, dtype=bool)
    if positive.all() or (~positive).all():
        raise ValueError("permutation requires both positive and control pairs")
    quantiles = np.unique(np.quantile(features[:, degree_column], [0, .2, .4, .6, .8, 1]))
    strata = np.digitize(features[:, degree_column], quantiles[1:-1], right=True)
    observed = features[positive].mean(axis=0) - features[~positive].mean(axis=0)
    rng = np.random.default_rng(seed)
    null = np.empty((samples, features.shape[1]), dtype=np.float64)
    for index in range(samples):
        permuted = positive.copy()
        for stratum in np.unique(strata):
            positions = np.flatnonzero(strata == stratum)
            permuted[positions] = rng.permutation(permuted[positions])
        null[index] = features[permuted].mean(axis=0) - features[~permuted].mean(axis=0)
    p_values = (1 + (np.abs(null) >= np.abs(observed)).sum(axis=0)) / (samples + 1)
    return {
        "samples": samples,
        "positive_pairs": int(positive.sum()),
        "control_pairs": int((~positive).sum()),
        "observed_mean_difference": observed.tolist(),
        "two_sided_p_value": p_values.tolist(),
    }

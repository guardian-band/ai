"""Small directed PrimeKG path features for cold-start signal probing."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import ttest_ind

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


def label_specific_degree_adjusted_enrichment(
    features: np.ndarray,
    targets: np.ndarray,
    label_names: Iterable[str],
    pair_positive: np.ndarray,
    *,
    degree_column: int = 6,
    feature_count: int = 8,
    minimum_positives: int = 20,
) -> dict[str, Any]:
    """Test label-specific enrichment only among observed-positive DDI pairs.

    For each label, graph features are residualized against graph degree and the
    number of *other* labels on the pair. This prevents the result from merely
    rediscovering the observed-positive/control split or overall label burden.
    """

    features = np.asarray(features[:, :feature_count], dtype=np.float32)
    targets = np.asarray(targets, dtype=bool)
    pair_positive = np.asarray(pair_positive, dtype=bool)
    labels = tuple(map(str, label_names))
    if targets.ndim != 2 or targets.shape[0] != features.shape[0] or targets.shape[1] != len(labels):
        raise ValueError("targets must align with feature rows and label names")
    if pair_positive.shape != (features.shape[0],):
        raise ValueError("pair_positive must align with feature rows")
    if not pair_positive.any():
        raise ValueError("label-specific enrichment requires observed-positive pairs")

    features = features[pair_positive]
    targets = targets[pair_positive]
    counts = targets.sum(axis=0)
    eligible = (counts >= minimum_positives) & ((len(targets) - counts) >= minimum_positives)
    total_label_burden = targets.sum(axis=1).astype(np.float64)
    observed = np.zeros((targets.shape[1], feature_count), dtype=np.float64)
    p_values = np.ones_like(observed)
    for label_index in np.flatnonzero(eligible):
        positive = targets[:, label_index]
        other_label_burden = total_label_burden - positive.astype(np.float64)
        design = np.column_stack(
            [
                np.ones(len(features)),
                np.log1p(features[:, degree_column].astype(np.float64)),
                other_label_burden,
            ]
        )
        coefficients, *_ = np.linalg.lstsq(design, features.astype(np.float64), rcond=None)
        adjusted_features = features.astype(np.float64) - design @ coefficients
        observed[label_index] = (
            adjusted_features[positive].mean(axis=0)
            - adjusted_features[~positive].mean(axis=0)
        )
        result = ttest_ind(
            adjusted_features[positive],
            adjusted_features[~positive],
            axis=0,
            equal_var=False,
            alternative="greater",
        )
        p_values[label_index] = np.nan_to_num(result.pvalue, nan=1.0)
    p_values[~eligible, :] = 1.0

    # Benjamini-Hochberg across every eligible label-feature hypothesis.
    flat = p_values.ravel()
    order = np.argsort(flat)
    ranked = flat[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q_flat = np.empty_like(adjusted)
    q_flat[order] = np.minimum(adjusted, 1.0)
    q_values = q_flat.reshape(p_values.shape)
    records = []
    for label_index, label in enumerate(labels):
        if not eligible[label_index]:
            continue
        for feature_index in range(feature_count):
            records.append(
                {
                    "label": label,
                    "feature_index": feature_index,
                    "positive_count": int(counts[label_index]),
                    "mean_difference": float(observed[label_index, feature_index]),
                    "p_value": float(p_values[label_index, feature_index]),
                    "q_value": float(q_values[label_index, feature_index]),
                }
            )
    records.sort(key=lambda row: (row["q_value"], -row["mean_difference"], row["label"]))
    enriched = [row for row in records if row["q_value"] < 0.05 and row["mean_difference"] > 0]
    return {
        "minimum_positives": minimum_positives,
        "comparison_population": "observed-positive DDI pairs only",
        "observed_positive_pairs": int(pair_positive.sum()),
        "eligible_labels": int(eligible.sum()),
        "hypotheses": len(records),
        "confounder_adjustment": "OLS residualization by log graph degree and other-label burden",
        "test": "one-sided Welch t-test",
        "fdr_method": "Benjamini-Hochberg",
        "significant_positive_hypotheses": len(enriched),
        "significant_positive_labels": len({row["label"] for row in enriched}),
        "top_enrichments": enriched[:25],
    }


def deduplicate_feature_columns(
    features: np.ndarray, feature_names: Iterable[str]
) -> tuple[np.ndarray, tuple[str, ...], dict[str, str]]:
    """Remove exactly equivalent feature columns and record their canonical name."""

    values = np.asarray(features)
    names = tuple(map(str, feature_names))
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("features must align with feature names")
    kept: list[int] = []
    aliases: dict[str, str] = {}
    for index, name in enumerate(names):
        duplicate = next(
            (prior for prior in kept if np.allclose(values[:, index], values[:, prior], rtol=0, atol=1e-8)),
            None,
        )
        if duplicate is None:
            kept.append(index)
        else:
            aliases[name] = names[duplicate]
    return values[:, kept], tuple(names[index] for index in kept), aliases


def label_specific_morgan_residual_enrichment(
    features: np.ndarray,
    targets: np.ndarray,
    probabilities: np.ndarray,
    label_names: Iterable[str],
    pair_positive: np.ndarray,
    feature_names: Iterable[str],
    *,
    degree_feature: str = "mean_protein_degree",
    minimum_positives: int = 20,
) -> dict[str, Any]:
    """Test whether graph features explain label residuals beyond frozen Morgan.

    Tests are limited to observed-positive DDI pairs. For every label-feature
    hypothesis, an OLS model predicts ``truth - Morgan probability`` while
    controlling for log graph degree and the number of other labels on the pair.
    Equivalent graph feature columns are removed before BH-FDR correction.
    """

    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=bool)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    pair_positive = np.asarray(pair_positive, dtype=bool)
    labels = tuple(map(str, label_names))
    names = tuple(map(str, feature_names))
    if targets.shape != probabilities.shape or targets.shape != (len(features), len(labels)):
        raise ValueError("targets, probabilities, features, and labels must align")
    if pair_positive.shape != (len(features),):
        raise ValueError("pair_positive must align with feature rows")

    features, names, aliases = deduplicate_feature_columns(features, names)
    if degree_feature not in names:
        canonical_degree = aliases.get(degree_feature)
        if canonical_degree is None:
            raise ValueError(f"missing degree feature: {degree_feature}")
        degree_feature = canonical_degree
    degree_column = names.index(degree_feature)
    connected_column = names.index("connected_within_three_hops")

    selected = pair_positive
    values = features[selected]
    truth = targets[selected]
    probs = probabilities[selected]
    residuals = truth.astype(np.float64) - probs
    total_label_burden = truth.sum(axis=1).astype(np.float64)
    counts = truth.sum(axis=0)
    eligible = (counts >= minimum_positives) & ((len(truth) - counts) >= minimum_positives)
    records: list[dict[str, Any]] = []
    for label_index in np.flatnonzero(eligible):
        other_label_burden = total_label_burden - truth[:, label_index].astype(np.float64)
        for feature_index, feature_name in enumerate(names):
            if feature_name in {degree_feature, "connected_within_three_hops"}:
                continue
            design = np.column_stack(
                [
                    np.ones(len(values)),
                    values[:, feature_index],
                    np.log1p(values[:, degree_column]),
                    other_label_burden,
                ]
            )
            if np.linalg.matrix_rank(design) < design.shape[1]:
                continue
            coefficients, *_ = np.linalg.lstsq(design, residuals[:, label_index], rcond=None)
            errors = residuals[:, label_index] - design @ coefficients
            degrees_of_freedom = len(values) - design.shape[1]
            variance = float(errors @ errors) / degrees_of_freedom
            covariance = variance * np.linalg.inv(design.T @ design)
            standard_error = float(np.sqrt(max(covariance[1, 1], 0.0)))
            coefficient = float(coefficients[1])
            statistic = coefficient / standard_error if standard_error > 0 else 0.0
            p_value = float(student_t.sf(statistic, degrees_of_freedom)) if standard_error > 0 else 1.0
            records.append(
                {
                    "label": labels[label_index],
                    "feature": feature_name,
                    "positive_count": int(counts[label_index]),
                    "coefficient": coefficient,
                    "p_value": p_value,
                }
            )

    if records:
        p_values = np.asarray([row["p_value"] for row in records])
        order = np.argsort(p_values)
        ranked = p_values[order]
        adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        q_values = np.empty_like(adjusted)
        q_values[order] = np.minimum(adjusted, 1.0)
        for row, q_value in zip(records, q_values, strict=True):
            row["q_value"] = float(q_value)
    records.sort(key=lambda row: (row.get("q_value", 1.0), -row["coefficient"], row["label"]))
    enriched = [row for row in records if row.get("q_value", 1.0) < 0.05 and row["coefficient"] > 0]

    connected = values[:, connected_column] > 0
    split_report = {}
    for name, mask in (("connected", connected), ("disconnected", ~connected)):
        split_residuals = residuals[mask]
        split_report[name] = {
            "pairs": int(mask.sum()),
            "mean_residual": float(split_residuals.mean()) if mask.any() else None,
            "mean_absolute_residual": float(np.abs(split_residuals).mean()) if mask.any() else None,
        }
    return {
        "comparison_population": "observed-positive DDI pairs only",
        "target": "truth_minus_frozen_morgan_probability",
        "confounder_adjustment": "OLS with log graph degree and other-label burden",
        "minimum_positives": minimum_positives,
        "eligible_labels": int(eligible.sum()),
        "deduplicated_feature_names": list(names),
        "equivalent_feature_aliases": aliases,
        "hypotheses": len(records),
        "fdr_method": "Benjamini-Hochberg",
        "significant_positive_hypotheses": len(enriched),
        "significant_positive_labels": len({row["label"] for row in enriched}),
        "connected_disconnected": split_report,
        "top_residual_enrichments": enriched[:25],
    }

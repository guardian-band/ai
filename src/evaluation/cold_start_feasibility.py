"""Leakage-safe, training-free diagnostics for cold-start DDI feasibility."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.data.primekg_typed_graph import LEAKAGE_RELATIONS, REQUIRED_COLUMNS


POSITIVE_STATUSES = {"observed_positive", "positive", "known_positive"}


def _normal(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def load_manifest_tables(manifest_path: str | Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, list[str]]:
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    artifacts = manifest.get("artifacts", {})

    def resolve(key: str) -> Path:
        raw = manifest.get(f"{key}_path") or artifacts.get(f"{key}_path")
        if not raw:
            raise ValueError(f"manifest does not define {key}_path")
        candidate = Path(raw)
        return candidate if candidate.is_absolute() else path.parent / candidate

    pairs = pd.read_parquet(resolve("pairs"))
    triples = pd.read_parquet(resolve("triples"))
    labels_payload = json.loads(resolve("labels").read_text())
    labels = [str(item["cui"]) for item in labels_payload["labels"]]
    return manifest, pairs, triples, labels


def _safe_primekg(frame: pd.DataFrame, target_label_ids: set[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"PrimeKG missing columns: {', '.join(sorted(missing))}")
    safe = frame[["x_id", "x_type", "relation", "y_id", "y_type"]].dropna().copy()
    for column in ("x_id", "y_id"):
        safe[column] = safe[column].astype(str).str.strip()
    for column in ("x_type", "relation", "y_type"):
        safe[column] = safe[column].map(_normal)
    relation_leakage = safe["relation"].isin(LEAKAGE_RELATIONS)
    label_leakage = safe["x_id"].isin(target_label_ids) | safe["y_id"].isin(target_label_ids)
    rejected = {
        "relation_edges": int(relation_leakage.sum()),
        "target_label_incident_edges": int((~relation_leakage & label_leakage).sum()),
    }
    safe = safe.loc[~relation_leakage & ~label_leakage].drop_duplicates(ignore_index=True)
    rejected["retained_edges"] = int(len(safe))
    return safe, rejected


def audit_safe_paths(
    primekg: pd.DataFrame,
    pairs: pd.DataFrame,
    target_label_ids: Iterable[str],
    *,
    splits: tuple[str, ...] = ("validation", "test"),
    max_hops: int = 3,
) -> dict[str, Any]:
    """Measure <=3-hop drug-pair connectivity after strict target leakage removal."""

    if max_hops != 3:
        raise ValueError("the audited protocol is fixed to max_hops=3")
    safe, rejected = _safe_primekg(primekg, set(map(str, target_label_ids)))
    adjacency: dict[tuple[str, str], dict[tuple[str, str], str]] = defaultdict(dict)
    drug_nodes: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in safe.itertuples(index=False):
        left = (str(row.x_type), str(row.x_id))
        right = (str(row.y_type), str(row.y_id))
        relation = str(row.relation)
        adjacency[left].setdefault(right, relation)
        adjacency[right].setdefault(left, relation)
        if "drug" in left[0]:
            drug_nodes[left[1]].add(left)
        if "drug" in right[0]:
            drug_nodes[right[1]].add(right)

    selected = pairs[pairs["split"].isin(splits)].copy()
    first_hop = {
        drug_id: {
            neighbor: relation
            for node in nodes
            for neighbor, relation in adjacency.get(node, {}).items()
        }
        for drug_id, nodes in drug_nodes.items()
    }

    def shortest_path(left_id: str, right_id: str):
        """Targeted <=3-hop search without expanding the whole three-hop ball."""

        left_nodes = drug_nodes.get(left_id, set())
        right_nodes = drug_nodes.get(right_id, set())
        for left_node in left_nodes:
            for right_node in right_nodes:
                relation = adjacency.get(left_node, {}).get(right_node)
                if relation is not None:
                    return 1, (relation,), ()

        left_neighbors = first_hop.get(left_id, {})
        right_neighbors = first_hop.get(right_id, {})
        common = left_neighbors.keys() & right_neighbors.keys()
        if common:
            middle = min(common)
            return (
                2,
                (left_neighbors[middle], right_neighbors[middle]),
                (middle[0],),
            )

        # A 3-hop path exists iff an edge joins a first-hop neighbor of each drug.
        # Iterate the lower-degree frontier to reduce work on hub-heavy PrimeKG.
        if len(left_neighbors) <= len(right_neighbors):
            outer, target = left_neighbors, right_neighbors
            reverse = False
        else:
            outer, target = right_neighbors, left_neighbors
            reverse = True
        target_nodes = target.keys()
        for first, edge_one in outer.items():
            bridge_candidates = adjacency.get(first, {}).keys() & target_nodes
            if not bridge_candidates:
                continue
            second = min(bridge_candidates)
            edge_two = adjacency[first][second]
            edge_three = target[second]
            if reverse:
                return 3, (edge_three, edge_two, edge_one), (second[0], first[0])
            return 3, (edge_one, edge_two, edge_three), (first[0], second[0])
        return None

    by_split: dict[str, Any] = {}
    for split in splits:
        subset = selected[selected["split"] == split]
        hops = Counter()
        relations = Counter()
        intermediate_types = Counter()
        for row in subset.itertuples(index=False):
            result = shortest_path(str(row.drug_a), str(row.drug_b))
            if result is None:
                hops["disconnected"] += 1
                continue
            distance, signature, intermediate = result
            hops[str(distance)] += 1
            relations.update(signature)
            intermediate_types.update(intermediate[: max(0, distance - 1)])
        connected = sum(hops[str(i)] for i in range(1, max_hops + 1))
        total = int(len(subset))
        by_split[split] = {
            "pairs": total,
            "connected_within_3_hops": connected,
            "coverage": connected / total if total else 0.0,
            "shortest_path_hops": dict(sorted(hops.items())),
            "relations_on_representative_shortest_paths": dict(relations.most_common()),
            "intermediate_node_types": dict(intermediate_types.most_common()),
        }
    return {"protocol": "strict_leakage_safe_undirected_shortest_path", "max_hops": 3, "filtering": rejected, "splits": by_split}


@dataclass(frozen=True)
class SimilarityResult:
    report: dict[str, Any]
    predictions: np.ndarray
    targets: np.ndarray


def evaluate_similarity_transfer(
    pairs: pd.DataFrame,
    triples: pd.DataFrame,
    labels: list[str],
    morgan: pd.DataFrame,
    *,
    split: str = "validation",
    neighbors: tuple[int, ...] = (1, 3, 5, 10),
) -> SimilarityResult:
    """Transfer train-pair label profiles through nearest Morgan neighbors."""

    fingerprints = {
        str(row.drugbank_id): np.asarray(row.morgan_fingerprint, dtype=np.float32)
        for row in morgan.itertuples(index=False)
    }
    label_index = {label: index for index, label in enumerate(labels)}
    pair_rows = {str(row.pair_id): row for row in pairs.itertuples(index=False)}
    pair_targets: dict[str, np.ndarray] = {
        pair_id: np.zeros(len(labels), dtype=np.float32) for pair_id in pair_rows
    }
    for row in triples.itertuples(index=False):
        if _normal(row.observation_status) in POSITIVE_STATUSES and str(row.label_cui) in label_index:
            pair_targets[str(row.pair_id)][label_index[str(row.label_cui)]] = 1.0

    train = pairs[pairs["split"] == "train"]
    target_pairs = pairs[pairs["split"] == split].sort_values("pair_id")
    train_drugs = sorted(set(train["drug_a"].astype(str)) | set(train["drug_b"].astype(str)))
    train_drugs = [drug for drug in train_drugs if drug in fingerprints]
    train_matrix = np.stack([fingerprints[drug] for drug in train_drugs]).astype(bool)
    train_pair_profiles: dict[tuple[str, str], np.ndarray] = {}
    for row in train.itertuples(index=False):
        key = tuple(sorted((str(row.drug_a), str(row.drug_b))))
        train_pair_profiles[key] = pair_targets[str(row.pair_id)]
    prevalence = np.stack(list(train_pair_profiles.values())).mean(axis=0)

    needed = sorted(set(target_pairs["drug_a"].astype(str)) | set(target_pairs["drug_b"].astype(str)))
    nearest: dict[str, list[tuple[str, float]]] = {}
    for drug in needed:
        if drug in train_drugs:
            nearest[drug] = [(drug, 1.0)]
            continue
        if drug not in fingerprints:
            nearest[drug] = []
            continue
        query = fingerprints[drug].astype(bool)
        intersection = np.logical_and(train_matrix, query).sum(axis=1)
        union = np.logical_or(train_matrix, query).sum(axis=1)
        similarities = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)
        order = np.argsort(-similarities, kind="stable")
        nearest[drug] = [(train_drugs[index], float(similarities[index])) for index in order]

    target_rows = list(target_pairs.itertuples(index=False))
    y_true = np.stack([pair_targets[str(row.pair_id)] for row in target_rows])
    pair_similarity = np.asarray(
        [
            min(
                nearest.get(str(row.drug_a), [("", 0.0)])[0][1] if nearest.get(str(row.drug_a)) else 0.0,
                nearest.get(str(row.drug_b), [("", 0.0)])[0][1] if nearest.get(str(row.drug_b)) else 0.0,
            )
            for row in target_rows
        ],
        dtype=np.float64,
    )
    candidates: dict[int, tuple[np.ndarray, float, float, float]] = {}
    for k in neighbors:
        rows: list[np.ndarray] = []
        transferred = 0
        for row in target_rows:
            left = nearest.get(str(row.drug_a), [])[:k]
            right = nearest.get(str(row.drug_b), [])[:k]
            weighted: list[tuple[float, np.ndarray]] = []
            for left_id, left_sim in left:
                for right_id, right_sim in right:
                    if left_id == right_id:
                        continue
                    profile = train_pair_profiles.get(tuple(sorted((left_id, right_id))))
                    if profile is not None:
                        weighted.append((left_sim * right_sim, profile))
            if weighted and sum(item[0] for item in weighted) > 0:
                transferred += 1
                denominator = sum(item[0] for item in weighted)
                rows.append(sum(weight * profile for weight, profile in weighted) / denominator)
            else:
                rows.append(prevalence)
        prediction = np.stack(rows)
        per_label = average_precision_score(y_true, prediction, average=None)
        macro = float(np.mean(per_label))
        micro = float(average_precision_score(y_true.ravel(), prediction.ravel()))
        candidates[int(k)] = (prediction, macro, micro, transferred / len(target_pairs))
    best_k = max(candidates, key=lambda item: (candidates[item][1], -item))
    prediction, macro, micro, coverage = candidates[best_k]
    bins = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0000001))
    similarity_strata: dict[str, Any] = {}
    for lower, upper in bins:
        mask = (pair_similarity >= lower) & (pair_similarity < upper)
        key = f"[{lower:.2f},{min(upper, 1.0):.2f}{']' if upper > 1 else ')'}"
        if not mask.any():
            similarity_strata[key] = {"pairs": 0, "macro_auprc": None, "micro_auprc": None}
            continue
        stratum_targets = y_true[mask]
        stratum_predictions = prediction[mask]
        similarity_strata[key] = {
            "pairs": int(mask.sum()),
            "macro_auprc": float(average_precision_score(stratum_targets, stratum_predictions, average="macro")),
            "micro_auprc": float(average_precision_score(stratum_targets.ravel(), stratum_predictions.ravel())),
        }
    report = {
        "protocol": "exploratory_validation_only_morgan_similarity_transfer",
        "split": split,
        "pair_count": int(len(target_pairs)),
        "best_k_selected_on_validation": int(best_k),
        "macro_auprc": macro,
        "micro_auprc": micro,
        "transfer_coverage": coverage,
        "candidates": {
            str(k): {"macro_auprc": value[1], "micro_auprc": value[2], "transfer_coverage": value[3]}
            for k, value in candidates.items()
        },
        "per_drug_nearest_train_similarity_including_known_endpoints": {
            "mean": float(np.mean([nearest[d][0][1] for d in needed if nearest.get(d)])),
            "minimum": float(np.min([nearest[d][0][1] for d in needed if nearest.get(d)])),
        },
        "per_pair_minimum_endpoint_similarity": {
            "mean": float(pair_similarity.mean()),
            "minimum": float(pair_similarity.min()),
            "definition": "minimum endpoint nearest-training Morgan similarity; known endpoints equal 1.0",
        },
        "performance_by_pair_nearest_train_similarity": similarity_strata,
    }
    return SimilarityResult(report, prediction, y_true)

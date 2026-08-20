"""Validation threshold selection with explicit objectives and aligned masks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


OBJECTIVES = {"max_f1", "max_recall_at_min_ppv"}


def _prepare_arrays(y_true: Any, y_prob: Any) -> tuple[np.ndarray, np.ndarray]:
    true_array = np.asarray(y_true, dtype=float)
    prob_array = np.asarray(y_prob, dtype=float)
    if true_array.shape != prob_array.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if np.isinf(true_array).any() or np.isinf(prob_array).any():
        raise ValueError("y_true and y_prob must be finite or NaN")
    if np.any((prob_array[~np.isnan(prob_array)] < 0.0) | (prob_array[~np.isnan(prob_array)] > 1.0)):
        raise ValueError("y_prob values must be in [0, 1]")

    valid_mask = ~np.isnan(true_array) & ~np.isnan(prob_array)
    true_valid = true_array[valid_mask]
    prob_valid = prob_array[valid_mask]
    if true_valid.size == 0:
        raise ValueError("threshold selection requires at least one aligned finite observation")
    if not np.isin(true_valid, [0.0, 1.0]).all():
        raise ValueError("y_true values must be binary")
    return true_valid, prob_valid


def _candidate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = y_prob >= threshold
    true_positive = float(np.sum(predictions & (y_true == 1.0)))
    false_positive = float(np.sum(predictions & (y_true == 0.0)))
    false_negative = float(np.sum(~predictions & (y_true == 1.0)))
    ppv = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2.0 * ppv * recall / (ppv + recall) if ppv + recall else 0.0
    return {
        "selected_ppv": float(ppv),
        "ppv": float(ppv),
        "recall": float(recall),
        "f1": float(f1),
        "positive_count": int(np.sum(y_true == 1.0)),
    }


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_ppv: float = 0.5,
    objective: str = "max_recall_at_min_ppv",
    *,
    mode: str | None = None,
) -> dict[str, Any]:
    """Select a threshold using an explicit objective.

    ``max_f1`` chooses the highest F1, breaking ties toward the higher
    threshold. ``max_recall_at_min_ppv`` first maximizes recall among candidates
    meeting ``min_ppv``, then PPV, then threshold; if none qualify it falls back
    to max-F1. NaNs are removed with one shared mask.
    """

    if mode is not None:
        objective = mode
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {sorted(OBJECTIVES)}")
    if not np.isfinite(min_ppv) or not 0.0 <= min_ppv <= 1.0:
        raise ValueError("min_ppv must be in [0, 1]")

    true_valid, prob_valid = _prepare_arrays(y_true, y_prob)
    sorted_indices = np.argsort(prob_valid, kind="mergesort")
    sorted_probabilities = prob_valid[sorted_indices]
    sorted_positives = true_valid[sorted_indices] == 1.0
    candidates = np.unique(
        np.concatenate(([0.0], sorted_probabilities, [1.0]))
    )
    candidate_indices = np.searchsorted(
        sorted_probabilities, candidates, side="left"
    )
    suffix_true_positives = np.empty(sorted_positives.size + 1, dtype=np.int64)
    suffix_true_positives[-1] = 0
    suffix_true_positives[:-1] = np.cumsum(
        sorted_positives[::-1], dtype=np.int64
    )[::-1]
    true_positives = suffix_true_positives[candidate_indices].astype(float)
    predicted_positives = (sorted_positives.size - candidate_indices).astype(float)
    false_positives = predicted_positives - true_positives
    total_positives = float(np.sum(sorted_positives))
    false_negatives = total_positives - true_positives
    ppv = np.divide(
        true_positives,
        predicted_positives,
        out=np.zeros_like(true_positives),
        where=predicted_positives > 0,
    )
    recall_denominator = true_positives + false_negatives
    recall = np.divide(
        true_positives,
        recall_denominator,
        out=np.zeros_like(true_positives),
        where=recall_denominator > 0,
    )
    f1_denominator = ppv + recall
    f1 = np.divide(
        2.0 * ppv * recall,
        f1_denominator,
        out=np.zeros_like(ppv),
        where=f1_denominator > 0,
    )

    def max_f1_index() -> int:
        # Candidates are ascending, so the final matching index implements
        # the existing higher-threshold tie break.
        return int(np.flatnonzero(f1 == np.max(f1))[-1])

    if objective == "max_f1":
        selected_index = max_f1_index()
        rule = "max_f1"
    else:
        eligible = np.flatnonzero(ppv >= min_ppv)
        if eligible.size:
            best_recall = np.max(recall[eligible])
            eligible = eligible[recall[eligible] == best_recall]
            best_ppv = np.max(ppv[eligible])
            eligible = eligible[ppv[eligible] == best_ppv]
            selected_index = int(eligible[-1])
            rule = f"max_recall_at_ppv_{float(min_ppv)}"
        else:
            selected_index = max_f1_index()
            rule = "max_f1"

    return {
        "threshold": float(candidates[selected_index]),
        "rule": rule,
        "selected_ppv": float(ppv[selected_index]),
        "ppv": float(ppv[selected_index]),
        "recall": float(recall[selected_index]),
        "f1": float(f1[selected_index]),
        "positive_count": int(total_positives),
    }


def select_thresholds(
    val_y_true: pd.DataFrame,
    val_y_prob: pd.DataFrame,
    min_ppv: float = 0.5,
) -> dict[str, dict[str, Any]]:
    """Select global and per-label thresholds from aligned validation tables."""

    if not isinstance(val_y_true, pd.DataFrame) or not isinstance(val_y_prob, pd.DataFrame):
        raise ValueError("val_y_true and val_y_prob must be pandas DataFrames")
    if list(val_y_true.columns) != list(val_y_prob.columns):
        raise ValueError("validation truth and probability columns must match")
    if val_y_true.shape != val_y_prob.shape:
        raise ValueError("validation truth and probability tables must have the same shape")

    global_result = find_best_threshold(
        val_y_true.to_numpy().ravel(),
        val_y_prob.to_numpy().ravel(),
        min_ppv=0.0,
        objective="max_f1",
    )
    global_threshold = global_result["threshold"]
    thresholds: dict[str, dict[str, Any]] = {}

    for label in val_y_true.columns:
        true_valid, prob_valid = _prepare_arrays(val_y_true[label].to_numpy(), val_y_prob[label].to_numpy())
        positive_count = int(np.sum(true_valid == 1.0))
        if positive_count < 10:
            selected_metrics = _candidate_metrics(true_valid, prob_valid, global_threshold)
            thresholds[str(label)] = {
                "threshold": float(global_threshold),
                "rule": "global_micro_f1_due_to_low_support",
                **selected_metrics,
            }
        else:
            selected = find_best_threshold(
                true_valid,
                prob_valid,
                min_ppv=min_ppv,
                objective="max_recall_at_min_ppv",
            )
            thresholds[str(label)] = selected
    return thresholds

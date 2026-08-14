import numpy as np
import pandas as pd
import pytest

from src.evaluation.thresholds import find_best_threshold, select_thresholds


def test_global_max_f1_does_not_fall_back_to_zero_threshold():
    result = find_best_threshold(
        np.array([1, 0, 1, 0]),
        np.array([0.9, 0.8, 0.4, 0.1]),
        objective="max_f1",
    )
    assert result["threshold"] == pytest.approx(0.4)
    assert result["rule"] == "max_f1"
    assert result["f1"] == pytest.approx(0.8)


def test_max_f1_ties_choose_higher_threshold():
    result = find_best_threshold(
        np.array([1, 0]), np.array([0.5, 0.5]), objective="max_f1"
    )
    assert result["threshold"] == pytest.approx(0.5)


def test_ppv_constrained_selection_and_tie_breaks():
    result = find_best_threshold(
        np.array([1, 1, 0, 0]),
        np.array([0.9, 0.8, 0.7, 0.1]),
        min_ppv=1.0,
        objective="max_recall_at_min_ppv",
    )
    assert result["threshold"] == pytest.approx(0.8)
    assert result["selected_ppv"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["positive_count"] == 2


def test_equal_recall_prefers_higher_ppv_then_threshold():
    result = find_best_threshold(
        np.array([1, 0, 1, 0]),
        np.array([0.8, 0.6, 0.7, 0.5]),
        min_ppv=0.0,
        objective="max_recall_at_min_ppv",
    )
    assert result["threshold"] == pytest.approx(0.7)
    assert result["selected_ppv"] == pytest.approx(1.0)


def test_no_ppv_candidate_falls_back_to_max_f1():
    result = find_best_threshold(
        np.array([1, 0]),
        np.array([0.4, 0.6]),
        min_ppv=1.0,
        objective="max_recall_at_min_ppv",
    )
    assert result["rule"] == "max_f1"
    assert "f1" in result


def test_select_thresholds_uses_global_max_f1_for_low_support():
    y_true = pd.DataFrame({"A": [1, 0, 0], "B": [0, 1, 0]})
    y_prob = pd.DataFrame({"A": [0.4, 0.3, 0.2], "B": [0.3, 0.9, 0.1]})
    selected = select_thresholds(y_true, y_prob, min_ppv=0.5)
    assert selected["A"]["rule"] == "global_micro_f1_due_to_low_support"
    assert selected["A"]["threshold"] == pytest.approx(0.4)
    assert selected["A"]["f1"] >= 0.0


def test_aligned_nan_mask_is_shared_between_labels_and_probabilities():
    result = find_best_threshold(
        np.array([1.0, np.nan, 0.0]),
        np.array([0.9, 0.1, 0.2]),
        objective="max_f1",
    )
    assert result["positive_count"] == 1
    assert result["selected_ppv"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "y_true, y_prob, message",
    [
        ([1, 0], [0.5], "same shape"),
        ([1, 2], [0.5, 0.4], "binary"),
        ([1, 0], [-0.1, 0.4], "[0, 1]"),
        ([1, 0], [np.inf, 0.4], "finite"),
    ],
)
def test_find_best_threshold_rejects_invalid_inputs(y_true, y_prob, message):
    with pytest.raises(ValueError, match=message):
        find_best_threshold(np.array(y_true), np.array(y_prob), objective="max_f1")


def test_select_thresholds_rejects_column_mismatch_and_invalid_probability():
    with pytest.raises(ValueError, match="columns"):
        select_thresholds(
            pd.DataFrame({"A": [1, 0]}), pd.DataFrame({"B": [0.5, 0.4]})
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        select_thresholds(
            pd.DataFrame({"A": [1, 0]}), pd.DataFrame({"A": [1.2, 0.4]})
        )

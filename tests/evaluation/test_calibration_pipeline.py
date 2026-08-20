import numpy as np
import pandas as pd
import pytest

from src.evaluation.calibration import (
    apply_temperature,
    calibrate_validation_logits,
    fit_temperature,
)


def test_apply_temperature_returns_sigmoid_probabilities_from_scaled_logits():
    logits = np.array([[-2.0, 0.0, 2.0]])
    calibrated = apply_temperature(logits, temperature=2.0)
    expected = 1.0 / (1.0 + np.exp(-logits / 2.0))
    assert np.allclose(calibrated, expected)
    assert np.all((calibrated >= 0.0) & (calibrated <= 1.0))


def test_fit_temperature_aligns_nan_labels_and_rejects_bad_shapes():
    logits = np.array([2.0, 1.0, -1.0, -2.0])
    labels = np.array([1.0, np.nan, 0.0, 0.0])
    assert 0.05 <= fit_temperature(logits, labels) <= 10.0

    with pytest.raises(ValueError, match="same shape"):
        fit_temperature(np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="finite"):
        fit_temperature(np.array([np.inf, 1.0]), np.array([1.0, 0.0]))


def test_degenerate_labels_use_identity_temperature():
    assert fit_temperature(np.array([-1.0, 1.0]), np.array([1.0, 1.0])) == 1.0


def test_degenerate_non_binary_labels_are_rejected():
    with pytest.raises(ValueError, match="binary"):
        fit_temperature(np.array([-1.0, 1.0]), np.array([2.0, 2.0]))


def test_validation_calibration_returns_probabilities_and_temperature():
    logits = pd.DataFrame({"C001": [-2.0, 0.0, 2.0], "C002": [1.0, -1.0, 0.5]})
    labels = pd.DataFrame({"C001": [0.0, 1.0, 1.0], "C002": [1.0, 0.0, 1.0]})
    probabilities, temperatures = calibrate_validation_logits(
        {"specific": logits}, {"specific": labels}
    )
    assert set(probabilities) == {"specific"}
    assert set(temperatures) == {"specific"}
    assert np.all((probabilities["specific"].to_numpy() >= 0.0) & (probabilities["specific"].to_numpy() <= 1.0))
    assert 0.05 <= temperatures["specific"] <= 10.0


def test_calibration_preserves_average_precision_ranking():
    from sklearn.metrics import average_precision_score

    logits = np.array([[-2.0, 0.0, 2.0]])
    labels = np.array([[0.0, 1.0, 1.0]])
    probabilities = apply_temperature(logits, 2.0)
    assert average_precision_score(labels.ravel(), logits.ravel()) == pytest.approx(
        average_precision_score(labels.ravel(), probabilities.ravel())
    )


def test_temperature_fit_improves_or_preserves_binary_nll():
    logits = np.tile(np.array([-4.0, -2.0, 2.0, 4.0]), 10_000)
    labels = np.tile(np.array([0.0, 0.0, 1.0, 1.0]), 10_000)
    temperature = fit_temperature(logits, labels)

    def nll(values):
        return np.mean(np.logaddexp(0.0, values) - labels * values)

    assert nll(logits / temperature) <= nll(logits) + 1e-12

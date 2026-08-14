"""Validation-only temperature scaling with explicit probability contracts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score


MIN_TEMPERATURE = 0.05
MAX_TEMPERATURE = 10.0


def _as_array(value: Any) -> np.ndarray:
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return value.to_numpy(dtype=float)
    return np.asarray(value, dtype=float)


def _validate_temperature(temperature: float) -> float:
    temperature = float(temperature)
    if not np.isfinite(temperature):
        raise ValueError("temperature must be finite")
    return float(np.clip(temperature, MIN_TEMPERATURE, MAX_TEMPERATURE))


def scale_logits(raw_logits: Any, temperature: float) -> np.ndarray:
    """Return raw logits divided by a validated temperature (still logits)."""

    logits = _as_array(raw_logits)
    if not np.isfinite(logits).all():
        raise ValueError("raw logits must be finite")
    return logits / _validate_temperature(temperature)


def apply_temperature(raw_logits: Any, temperature: float) -> np.ndarray:
    """Return calibrated probabilities: ``sigmoid(raw_logits / temperature)``."""

    scaled = scale_logits(raw_logits, temperature)
    probabilities = 1.0 / (1.0 + np.exp(-scaled))
    return np.clip(probabilities, 0.0, 1.0)


class TemperatureScaler(nn.Module):
    """Torch helper used internally to optimize a scalar temperature."""

    def __init__(self):
        super().__init__()
        self.log_temp = nn.Parameter(torch.zeros(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = torch.clamp(torch.exp(self.log_temp), MIN_TEMPERATURE, MAX_TEMPERATURE)
        return logits / temperature


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit one validation temperature from aligned raw logits and binary labels.

    NaN labels are excluded using one shared mask. Logits must be finite and
    shape-compatible with labels. Degenerate labels use the identity scale.
    """

    logits_array = _as_array(logits)
    labels_array = _as_array(labels)
    if logits_array.shape != labels_array.shape:
        raise ValueError("logits and labels must have the same shape")
    if not np.isfinite(logits_array).all():
        raise ValueError("logits must be finite")
    if np.isinf(labels_array).any():
        raise ValueError("labels must be finite or NaN")

    valid_mask = ~np.isnan(labels_array)
    valid_logits = logits_array[valid_mask]
    valid_labels = labels_array[valid_mask]
    if not np.isin(valid_labels, [0.0, 1.0]).all():
        raise ValueError("labels must be binary values or NaN")
    if valid_logits.size == 0 or np.unique(valid_labels).size < 2:
        return 1.0

    t_logits = torch.tensor(valid_logits, dtype=torch.float32)
    t_labels = torch.tensor(valid_labels, dtype=torch.float32)
    bce = nn.BCEWithLogitsLoss()
    initial_nll = float(bce(t_logits, t_labels).item())
    initial_ap = float(average_precision_score(valid_labels, valid_logits))

    scaler = TemperatureScaler()
    optimizer = optim.LBFGS(scaler.parameters(), lr=0.01, max_iter=100)

    def eval_closure():
        optimizer.zero_grad()
        loss = bce(scaler(t_logits), t_labels)
        loss.backward()
        return loss

    optimizer.step(eval_closure)
    fitted_temperature = _validate_temperature(torch.exp(scaler.log_temp).item())
    scaled_logits = valid_logits / fitted_temperature
    final_nll = float(bce(torch.tensor(scaled_logits), t_labels).item())
    final_ap = float(average_precision_score(valid_labels, scaled_logits))

    # Temperature scaling is monotonic, so ranking/AP must remain unchanged.
    if final_nll < initial_nll and abs(final_ap - initial_ap) <= 1e-10:
        return fitted_temperature
    return 1.0


def calibrate_validation_logits(
    val_logits: dict[str, Any], val_labels: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, float]]:
    """Fit validation temperatures and return calibrated probabilities.

    The first return value contains probabilities in [0, 1], not logits. The
    second contains scalar temperatures. Test data must not be supplied here.
    """

    probabilities: dict[str, Any] = {}
    temperatures: dict[str, float] = {}
    for level in sorted(set(val_logits).intersection(val_labels)):
        logits_value = val_logits[level]
        labels_value = val_labels[level]
        logits_array = _as_array(logits_value)
        labels_array = _as_array(labels_value)
        temperature = fit_temperature(logits_array, labels_array)
        calibrated_array = apply_temperature(logits_array, temperature)
        if isinstance(logits_value, pd.DataFrame):
            calibrated_value = pd.DataFrame(
                calibrated_array, index=logits_value.index, columns=logits_value.columns
            )
        elif isinstance(logits_value, pd.Series):
            calibrated_value = pd.Series(calibrated_array, index=logits_value.index, name=logits_value.name)
        else:
            calibrated_value = calibrated_array
        probabilities[level] = calibrated_value
        temperatures[level] = float(temperature)
    return probabilities, temperatures


def calibrate_logits(val_logits: dict[str, Any], val_labels: dict[str, Any]):
    """Compatibility wrapper; returns calibrated probabilities, never logits."""

    return calibrate_validation_logits(val_logits, val_labels)

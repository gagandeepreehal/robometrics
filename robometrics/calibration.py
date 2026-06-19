"""Calibration metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from robometrics.geometry import as_1d_array


def calibration_error(
    confidences: ArrayLike,
    outcomes: ArrayLike,
    n_bins: int = 10,
) -> float:
    """Return expected calibration error for binary outcomes and confidences."""
    confidence_arr = as_1d_array(confidences, name="confidences")
    outcome_arr = as_1d_array(outcomes, name="outcomes")
    if confidence_arr.shape != outcome_arr.shape:
        raise ValueError("confidences and outcomes must have the same shape")
    if not np.all((confidence_arr >= 0.0) & (confidence_arr <= 1.0)):
        raise ValueError("confidences must be in [0, 1]")
    if not np.all((outcome_arr == 0.0) | (outcome_arr == 1.0)):
        raise ValueError("outcomes must contain only boolean or 0/1 values")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    total = float(confidence_arr.shape[0])
    ece = 0.0
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    for index in range(int(n_bins)):
        lower = edges[index]
        upper = edges[index + 1]
        if index == int(n_bins) - 1:
            mask = (confidence_arr >= lower) & (confidence_arr <= upper)
        else:
            mask = (confidence_arr >= lower) & (confidence_arr < upper)
        if not np.any(mask):
            continue
        accuracy = float(np.mean(outcome_arr[mask]))
        confidence = float(np.mean(confidence_arr[mask]))
        ece += float(np.sum(mask)) / total * abs(accuracy - confidence)
    return float(ece)

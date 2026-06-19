"""Confidence calibration metrics."""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from robometrics.geometry import as_boolean_mask, as_numeric_array


def calibration_error(
    confidences: ArrayLike,
    correctness: Optional[ArrayLike] = None,
    *,
    outcomes: Optional[ArrayLike] = None,
    n_bins: int = 10,
) -> float:
    """Return Expected Calibration Error for confidence predictions."""
    if correctness is None and outcomes is None:
        raise ValueError("calibration_error requires correctness or outcomes")
    if correctness is not None and outcomes is not None:
        raise ValueError("provide either correctness or outcomes, not both")
    labels = correctness if correctness is not None else outcomes
    assert labels is not None

    confidence_arr = as_numeric_array(confidences, name="confidences").reshape(-1)
    correctness_arr = as_boolean_mask(labels, name="correctness").reshape(-1)
    if confidence_arr.shape != correctness_arr.shape:
        raise ValueError("confidences and correctness must have the same shape")
    if confidence_arr.size == 0:
        raise ValueError("confidences must contain at least one value")
    if np.any((confidence_arr < 0.0) | (confidence_arr > 1.0)):
        raise ValueError("confidences must be in [0, 1]")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    total = float(confidence_arr.size)
    error = 0.0
    correctness_float = correctness_arr.astype(np.float64)
    for bin_index in range(int(n_bins)):
        lower = edges[bin_index]
        upper = edges[bin_index + 1]
        if bin_index == int(n_bins) - 1:
            member_mask = (confidence_arr >= lower) & (confidence_arr <= upper)
        else:
            member_mask = (confidence_arr >= lower) & (confidence_arr < upper)
        if not np.any(member_mask):
            continue
        bin_accuracy = float(np.mean(correctness_float[member_mask]))
        bin_confidence = float(np.mean(confidence_arr[member_mask]))
        bin_weight = float(np.count_nonzero(member_mask)) / total
        error += bin_weight * abs(bin_accuracy - bin_confidence)
    return float(error)

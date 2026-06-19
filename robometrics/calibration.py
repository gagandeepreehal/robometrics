"""Confidence calibration metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from robometrics.geometry import as_boolean_mask, as_numeric_array


def calibration_error(
    confidences: ArrayLike,
    correctness: ArrayLike,
    *,
    n_bins: int = 10,
) -> float:
    """Return Expected Calibration Error for confidence predictions.

    Formula:
        Flatten confidence and correctness arrays. Partition confidence values
        into ``n_bins`` uniform bins over ``[0, 1]``. For every non-empty bin,
        compute ``abs(mean(correctness_bin) - mean(confidence_bin))`` and weight
        it by ``count_bin / total_count``. The result is the sum of these
        weighted gaps.

    Inputs:
        ``confidences`` must be finite probabilities in ``[0, 1]``.
        ``correctness`` must be boolean or 0/1 labels with the same shape.

    Output:
        A unitless non-negative error where lower is better. Perfect
        calibration returns ``0.0`` for the chosen bins.
    """
    confidence_arr = as_numeric_array(confidences, name="confidences").reshape(-1)
    correctness_arr = as_boolean_mask(correctness, name="correctness").reshape(-1)
    if confidence_arr.shape != correctness_arr.shape:
        raise ValueError("confidences and correctness must have the same shape")
    if confidence_arr.size == 0:
        raise ValueError("confidences must contain at least one value")
    if np.any((confidence_arr < 0.0) | (confidence_arr > 1.0)):
        raise ValueError("confidences must be in [0, 1]")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    bin_indices = np.minimum((confidence_arr * n_bins).astype(np.int64), n_bins - 1)
    total = float(confidence_arr.size)
    error = 0.0
    correctness_float = correctness_arr.astype(np.float64)
    for bin_index in range(n_bins):
        member_mask = bin_indices == bin_index
        if not np.any(member_mask):
            continue
        bin_accuracy = float(np.mean(correctness_float[member_mask]))
        bin_confidence = float(np.mean(confidence_arr[member_mask]))
        bin_weight = float(np.count_nonzero(member_mask)) / total
        error += bin_weight * abs(bin_accuracy - bin_confidence)
    return float(error)

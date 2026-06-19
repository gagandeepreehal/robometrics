"""Prediction diversity metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from robometrics.geometry import as_prediction_set


def trajectory_diversity(predictions: ArrayLike) -> float:
    """Return mean pairwise ADE between predicted trajectory modes."""
    pred_arr = as_prediction_set(predictions)
    if pred_arr.shape[0] < 2:
        return 0.0

    distances = []
    for left in range(pred_arr.shape[0]):
        for right in range(left + 1, pred_arr.shape[0]):
            pairwise = np.linalg.norm(pred_arr[left] - pred_arr[right], axis=1)
            distances.append(float(np.mean(pairwise)))
    return float(np.mean(distances))

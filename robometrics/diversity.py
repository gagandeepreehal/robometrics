"""Behavioral and trajectory diversity metrics."""

from __future__ import annotations

from math import sqrt
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robometrics.geometry import FloatArray, as_numeric_array, as_prediction_set


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


def behavioral_diversity(
    behaviors: ArrayLike,
    *,
    max_pairs: Optional[int] = 10_000,
    normalize: bool = False,
) -> float:
    """Return mean pairwise distance between unique behaviors."""
    matrix = _as_behavior_matrix(behaviors)
    unique_behaviors = np.unique(matrix, axis=0)
    count = unique_behaviors.shape[0]
    if count < 2:
        return 0.0

    if max_pairs is not None and max_pairs <= 0:
        raise ValueError("max_pairs must be positive or None")

    total_pairs = count * (count - 1) // 2
    pair_indices: NDArray[np.int64]
    if max_pairs is None or total_pairs <= max_pairs:
        pair_indices = np.arange(total_pairs, dtype=np.int64)
    else:
        pair_indices = np.asarray(
            np.linspace(0, total_pairs - 1, num=max_pairs, dtype=np.int64),
            dtype=np.int64,
        )

    distance_sum = 0.0
    scale = sqrt(float(unique_behaviors.shape[1])) if normalize else 1.0
    for condensed_index in pair_indices:
        left_index, right_index = _pair_from_condensed_index(int(condensed_index), count)
        distance = float(
            np.linalg.norm(unique_behaviors[left_index] - unique_behaviors[right_index])
        )
        distance_sum += distance / scale

    return float(distance_sum / float(pair_indices.shape[0]))


def _as_behavior_matrix(behaviors: ArrayLike) -> FloatArray:
    arr = as_numeric_array(behaviors, name="behaviors")
    if arr.ndim == 2:
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError("behaviors must have non-empty sample and feature dimensions")
        return arr
    if arr.ndim == 3:
        if arr.shape[0] == 0 or arr.shape[1] == 0 or arr.shape[2] == 0:
            raise ValueError("behaviors must have non-empty batch, time, and feature dimensions")
        return np.asarray(arr.reshape(arr.shape[0], -1), dtype=np.float64)
    raise ValueError("behaviors must be an NxD or NxTxD array")


def _pair_from_condensed_index(index: int, count: int) -> tuple[int, int]:
    left = int((2 * count - 1 - sqrt((2 * count - 1) ** 2 - 8 * index)) // 2)
    pairs_before_left = left * (2 * count - left - 1) // 2
    right = index - pairs_before_left + left + 1
    return left, right

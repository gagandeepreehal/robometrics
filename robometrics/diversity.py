"""Behavioral diversity metrics."""

from __future__ import annotations

from math import sqrt
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from robometrics.geometry import FloatArray, as_numeric_array


def behavioral_diversity(
    behaviors: ArrayLike,
    *,
    max_pairs: Optional[int] = 10_000,
    normalize: bool = False,
) -> float:
    """Return mean pairwise distance between unique behaviors.

    Formula:
        Accept behavior embeddings shaped ``NxD`` or trajectory/action batches
        shaped ``NxTxD``. Flatten each behavior to one vector, remove duplicate
        rows, and compute the mean Euclidean distance over unique behavior
        pairs. If ``normalize`` is true, divide each distance by
        ``sqrt(feature_count)``.

    Inputs:
        Finite numeric behavior arrays. ``max_pairs`` optionally caps the
        number of pair distances using deterministic, evenly spaced pair
        indices; no randomness is used.

    Output:
        A non-negative diversity score where higher means more diverse.
        Fewer than two unique behaviors return ``0.0``.
    """
    matrix = _as_behavior_matrix(behaviors)
    unique_behaviors = np.unique(matrix, axis=0)
    count = unique_behaviors.shape[0]
    if count < 2:
        return 0.0

    if max_pairs is not None and max_pairs <= 0:
        raise ValueError("max_pairs must be positive or None")

    total_pairs = count * (count - 1) // 2
    if max_pairs is None or total_pairs <= max_pairs:
        pair_indices = np.arange(total_pairs, dtype=np.int64)
    else:
        pair_indices = np.linspace(0, total_pairs - 1, num=max_pairs, dtype=np.int64)

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
    # Pair order matches nested loops: (0, 1), (0, 2), ..., (1, 2), ...
    left = int((2 * count - 1 - sqrt((2 * count - 1) ** 2 - 8 * index)) // 2)
    pairs_before_left = left * (2 * count - left - 1) // 2
    right = index - pairs_before_left + left + 1
    return left, right

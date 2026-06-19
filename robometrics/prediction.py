"""Multi-modal trajectory prediction metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from robometrics.geometry import (
    FloatArray,
    as_prediction_set,
    as_trajectory,
    require_same_time_and_dim,
)


def min_ade(predictions: ArrayLike, gt: ArrayLike) -> float:
    """Return the minimum average displacement error over predicted modes."""
    errors = _ade_by_mode(predictions, gt)
    return float(np.min(errors))


def min_fde(predictions: ArrayLike, gt: ArrayLike) -> float:
    """Return the minimum final displacement error over predicted modes."""
    errors = _fde_by_mode(predictions, gt)
    return float(np.min(errors))


def miss_rate(predictions: ArrayLike, gt: ArrayLike, threshold: float) -> float:
    """Return a per-sample miss indicator as 0.0 or 1.0.

    Average this value across a dataset to compute a conventional miss rate.
    """
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be a non-negative finite value")
    return float(min_fde(predictions, gt) > threshold)


def topk_trajectory_error(predictions: ArrayLike, gt: ArrayLike, k: int) -> float:
    """Return best ADE among the first k predictions, assumed confidence-ranked."""
    pred_arr = as_prediction_set(predictions)
    if k <= 0:
        raise ValueError("k must be positive")
    if k > pred_arr.shape[0]:
        raise ValueError("k cannot exceed the number of predicted trajectories")
    errors = _ade_by_mode(pred_arr[:k], gt)
    return float(np.min(errors))


def _ade_by_mode(predictions: ArrayLike, gt: ArrayLike) -> FloatArray:
    pred_arr = as_prediction_set(predictions)
    gt_arr = as_trajectory(gt, name="gt")
    require_same_time_and_dim(pred_arr, gt_arr)
    distances = np.linalg.norm(pred_arr - gt_arr[None, :, :], axis=2)
    return np.asarray(np.mean(distances, axis=1), dtype=np.float64)


def _fde_by_mode(predictions: ArrayLike, gt: ArrayLike) -> FloatArray:
    pred_arr = as_prediction_set(predictions)
    gt_arr = as_trajectory(gt, name="gt")
    require_same_time_and_dim(pred_arr, gt_arr)
    distances = np.linalg.norm(pred_arr[:, -1, :] - gt_arr[-1][None, :], axis=1)
    return np.asarray(distances, dtype=np.float64)

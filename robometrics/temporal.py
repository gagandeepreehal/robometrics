"""Temporal drift and control-sequence metrics."""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from robometrics.geometry import (
    FloatArray,
    as_batch_time_array,
    as_numeric_array,
    require_same_shape,
    validate_positive,
)


def temporal_drift(predicted: ArrayLike, reference: ArrayLike) -> float:
    """Return the mean slope of L2 error over time.

    Formula:
        For ``predicted`` and ``reference`` with shape ``TxD`` or ``BxTxD``,
        compute ``e[t] = ||predicted[t] - reference[t]||_2`` per batch item.
        The metric is the least-squares slope of ``e`` against timestep index,
        averaged across the batch.

    Inputs:
        Numeric finite arrays with identical shape. Units match the state
        coordinates per timestep index.

    Output:
        A scalar drift rate. ``0.0`` means no temporal growth in error,
        positive values mean error grows over time, and negative values mean it
        shrinks. A single timestep returns ``0.0`` because slope is undefined.
    """
    errors = _l2_errors(predicted, reference)
    slopes = _linear_slopes(errors)
    return float(np.mean(slopes))


def action_jerk(actions: ArrayLike, dt: float = 1.0) -> float:
    """Return mean squared second finite difference of an action sequence.

    Formula:
        For ``actions`` shaped ``TxD`` or ``BxTxD``, compute
        ``diff(actions, n=2) / dt**2`` over time, take the L2 norm per step,
        square it, then average over all measured batch timesteps.

    Inputs:
        Numeric finite action or control values. ``dt`` is seconds and must be
        positive.

    Output:
        A non-negative penalty in squared ``(action-units / dt^2)^2``. The
        exact SI unit depends on what the actions represent: squared m²/s⁴ for
        position-valued actions, m²/s⁶ for velocity-valued actions, etc. Lower
        is smoother. Constant and linear action sequences return ``0.0``
        because their second finite difference is zero. Fewer than three
        timesteps also return ``0.0``.
    """
    action_arr = as_batch_time_array(actions, name="actions")
    timestep = validate_positive(float(dt), name="dt")
    if action_arr.shape[1] < 3:
        return 0.0

    second_difference = np.diff(action_arr, n=2, axis=1) / (timestep * timestep)
    magnitudes = np.linalg.norm(second_difference, axis=2)
    return float(np.mean(np.square(magnitudes)))


def control_smoothness(actions: ArrayLike, dt: float = 1.0) -> float:
    """Return a bounded control smoothness score.

    Formula:
        ``1 / (1 + action_jerk(actions, dt))``.

    Inputs:
        Numeric finite action or control arrays shaped ``TxD`` or ``BxTxD``.
        ``dt`` is seconds and must be positive.

    Output:
        A dimensionless score in ``(0, 1]`` where higher is smoother.
        Constant and linear controls return the best score, ``1.0``.
    """
    penalty = action_jerk(actions, dt=dt)
    return float(1.0 / (1.0 + penalty))


def long_horizon_drift(predicted: ArrayLike, reference: ArrayLike) -> float:
    """Return later-weighted mean L2 rollout error.

    Formula:
        Compute per-timestep L2 errors as in ``temporal_drift``. Weight horizon
        step ``t`` by ``t + 1`` and return the weighted mean, averaged across
        batch items: ``sum((t + 1) * e[t]) / sum(t + 1)``.

    Inputs:
        Numeric finite rollout arrays shaped ``TxD`` or ``BxTxD`` with
        identical shape.

    Output:
        A non-negative error in state units. Perfect rollouts return ``0.0``.
        The same mean error is scored worse when concentrated later in the
        horizon.
    """
    errors = _l2_errors(predicted, reference)
    weights = np.arange(1, errors.shape[1] + 1, dtype=np.float64)
    weighted = errors @ weights / float(np.sum(weights))
    return float(np.mean(weighted))


def compounding_error_index(
    errors_or_predicted: ArrayLike,
    reference: Optional[ArrayLike] = None,
    *,
    epsilon: float = 1e-12,
) -> float:
    """Return a normalized non-negative error-growth index.

    Formula:
        If ``reference`` is provided, first compute per-timestep L2 errors from
        ``errors_or_predicted`` and ``reference`` shaped ``TxD`` or ``BxTxD``.
        Otherwise treat ``errors_or_predicted`` as per-timestep errors shaped
        ``T`` or ``BxT``. For each batch curve, compute
        ``max(0, final_error - initial_error) / (mean_error + epsilon)`` and
        average across the batch.

    Inputs:
        Finite non-negative error curves, or finite predicted/reference state
        arrays with identical shape. ``epsilon`` must be positive.

    Output:
        A dimensionless penalty. ``0.0`` means error did not increase from the
        first to final timestep; larger values mean stronger compounding.
    """
    eps = validate_positive(float(epsilon), name="epsilon")
    if reference is None:
        errors = _as_error_curves(errors_or_predicted)
    else:
        errors = _l2_errors(errors_or_predicted, reference)

    growth = np.maximum(errors[:, -1] - errors[:, 0], 0.0)
    mean_error = np.mean(errors, axis=1)
    values = growth / (mean_error + eps)
    return float(np.mean(values))


def _l2_errors(predicted: ArrayLike, reference: ArrayLike) -> FloatArray:
    predicted_arr = as_batch_time_array(predicted, name="predicted")
    reference_arr = as_batch_time_array(reference, name="reference")
    require_same_shape(predicted_arr, reference_arr, "predicted", "reference")
    return np.asarray(np.linalg.norm(predicted_arr - reference_arr, axis=2), dtype=np.float64)


def _linear_slopes(errors: FloatArray) -> FloatArray:
    if errors.shape[1] < 2:
        return np.zeros(errors.shape[0], dtype=np.float64)

    time = np.arange(errors.shape[1], dtype=np.float64)
    centered_time = time - float(np.mean(time))
    denominator = float(np.sum(np.square(centered_time)))
    centered_errors = errors - np.mean(errors, axis=1, keepdims=True)
    slopes = centered_errors @ centered_time / denominator
    return np.asarray(slopes, dtype=np.float64)


def _as_error_curves(errors: ArrayLike) -> FloatArray:
    arr = as_numeric_array(errors, name="errors_or_predicted")
    if arr.ndim == 1:
        if arr.shape[0] == 0:
            raise ValueError("errors_or_predicted must contain at least one timestep")
        curves = arr[None, :]
    elif arr.ndim == 2:
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError("errors_or_predicted must have non-empty batch and time axes")
        curves = arr
    else:
        raise ValueError("errors_or_predicted must be T or BxT when reference is omitted")

    if np.any(curves < 0.0):
        raise ValueError("errors_or_predicted must contain non-negative errors")
    return np.asarray(curves, dtype=np.float64)

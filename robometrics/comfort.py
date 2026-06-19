"""Planning and control smoothness metrics."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from robometrics.geometry import FloatArray, as_trajectory, validate_positive, vector_norms


def acceleration(traj: ArrayLike, dt: float) -> FloatArray:
    """Return approximate acceleration vectors from a position trajectory."""
    traj_arr = as_trajectory(traj, name="traj")
    timestep = validate_positive(float(dt), name="dt")
    return _acceleration_from_array(traj_arr, timestep)


def _acceleration_from_array(traj_arr: FloatArray, timestep: float) -> FloatArray:
    if traj_arr.shape[0] < 3:
        return np.zeros_like(traj_arr, dtype=np.float64)
    velocity = _gradient(traj_arr, timestep)
    return _gradient(velocity, timestep)


def jerk(traj: ArrayLike, dt: float) -> FloatArray:
    """Return approximate jerk vectors from a position trajectory."""
    traj_arr = as_trajectory(traj, name="traj")
    timestep = validate_positive(float(dt), name="dt")
    accel = _acceleration_from_array(traj_arr, timestep)
    if accel.shape[0] < 3:
        return np.zeros_like(accel, dtype=np.float64)
    return _gradient(accel, timestep)


def jerk_cost(traj: ArrayLike, dt: float) -> float:
    """Return mean squared jerk magnitude."""
    jerk_values = jerk(traj, dt)
    return float(np.mean(np.square(vector_norms(jerk_values))))


def max_acceleration(traj: ArrayLike, dt: float) -> float:
    """Return maximum acceleration magnitude."""
    accel = acceleration(traj, dt)
    return float(np.max(vector_norms(accel)))


def max_deceleration(traj: ArrayLike, dt: float) -> float:
    """Return maximum longitudinal deceleration magnitude."""
    traj_arr = as_trajectory(traj, name="traj")
    timestep = validate_positive(float(dt), name="dt")
    if traj_arr.shape[0] < 3:
        return 0.0

    velocity = _gradient(traj_arr, timestep)
    accel = _gradient(velocity, timestep)
    speeds = vector_norms(velocity)
    moving = speeds > 1e-12
    if not np.any(moving):
        return 0.0

    velocity_unit = np.zeros_like(velocity, dtype=np.float64)
    velocity_unit[moving] = velocity[moving] / speeds[moving, None]
    longitudinal_accel = np.sum(accel * velocity_unit, axis=1)
    deceleration = np.maximum(-longitudinal_accel, 0.0)
    return float(np.max(deceleration))


def smoothness_score(traj: ArrayLike, dt: float) -> float:
    """Return ``1 / (1 + log1p(cost))`` for dimensionless third-difference cost.

    The score is 1.0 for trajectories with no measurable third finite
    difference. At least four points are required to measure that difference,
    so shorter trajectories return 1.0 after normal input validation.
    """
    cost = _dimensionless_jerk_cost(traj, dt)
    return float(1.0 / (1.0 + np.log1p(cost)))


def _gradient(values: FloatArray, dt: float) -> FloatArray:
    edge_order: Literal[1, 2] = 2 if values.shape[0] > 2 else 1
    gradient = np.gradient(values, dt, axis=0, edge_order=edge_order)
    return np.asarray(gradient, dtype=np.float64)


def _dimensionless_jerk_cost(traj: ArrayLike, dt: float) -> float:
    traj_arr = as_trajectory(traj, name="traj")
    validate_positive(float(dt), name="dt")
    if traj_arr.shape[0] < 4:
        return 0.0

    third_difference = np.diff(traj_arr, n=3, axis=0)
    return float(np.mean(np.square(vector_norms(third_difference))))

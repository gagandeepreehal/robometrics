"""Planning and control smoothness metrics."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from robotmetrics.geometry import FloatArray, as_trajectory, validate_positive, vector_norms


def acceleration(traj: ArrayLike, dt: float) -> FloatArray:
    """Return approximate acceleration vectors from a position trajectory."""
    traj_arr = as_trajectory(traj, name="traj")
    timestep = validate_positive(float(dt), name="dt")
    if traj_arr.shape[0] < 3:
        return np.zeros_like(traj_arr, dtype=np.float64)
    velocity = _gradient(traj_arr, timestep)
    return _gradient(velocity, timestep)


def jerk(traj: ArrayLike, dt: float) -> FloatArray:
    """Return approximate jerk vectors from a position trajectory."""
    accel = acceleration(traj, dt)
    if accel.shape[0] < 3:
        return np.zeros_like(accel, dtype=np.float64)
    return _gradient(accel, validate_positive(float(dt), name="dt"))


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
    """Return a bounded smoothness score where 1.0 is smoother and 0.0 is worse."""
    return float(1.0 / (1.0 + jerk_cost(traj, dt)))


def _gradient(values: FloatArray, dt: float) -> FloatArray:
    edge_order: Literal[1, 2] = 2 if values.shape[0] > 2 else 1
    gradient = np.gradient(values, dt, axis=0, edge_order=edge_order)
    return np.asarray(gradient, dtype=np.float64)

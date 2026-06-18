"""Physical consistency and dynamic feasibility metrics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from robometrics.comfort import acceleration, jerk
from robometrics.geometry import (
    FloatArray,
    as_trajectory,
    validate_nonnegative,
    validate_positive,
    vector_norms,
)
from robometrics.schemas import MetricResult
from robometrics.trajectory import curvature

_DYNAMIC_FEASIBILITY_CONSTRAINTS = frozenset(
    {"max_speed", "max_accel", "max_jerk", "max_curvature"}
)


def speed_profile(traj: ArrayLike, dt: float) -> FloatArray:
    """Return speed magnitude at each trajectory point."""
    traj_arr = as_trajectory(traj, name="traj")
    timestep = validate_positive(float(dt), name="dt")
    if traj_arr.shape[0] == 1:
        return np.zeros(1, dtype=np.float64)
    edge_order: Literal[1, 2] = 2 if traj_arr.shape[0] > 2 else 1
    velocity = np.gradient(traj_arr, timestep, axis=0, edge_order=edge_order)
    return vector_norms(velocity.astype(np.float64))


def acceleration_limits_violated(traj: ArrayLike, dt: float, max_accel: float) -> MetricResult:
    """Return a MetricResult describing whether acceleration exceeds max_accel."""
    limit = validate_nonnegative(float(max_accel), name="max_accel")
    max_observed = float(np.max(vector_norms(acceleration(traj, dt))))
    passed = max_observed <= limit
    return MetricResult(
        name="acceleration_limits_violated",
        value=max_observed,
        unit="m/s^2",
        passed=passed,
        threshold=limit,
        metadata={"violated": not passed},
    )


def jerk_limits_violated(traj: ArrayLike, dt: float, max_jerk: float) -> MetricResult:
    """Return a MetricResult describing whether jerk exceeds max_jerk."""
    limit = validate_nonnegative(float(max_jerk), name="max_jerk")
    max_observed = float(np.max(vector_norms(jerk(traj, dt))))
    passed = max_observed <= limit
    return MetricResult(
        name="jerk_limits_violated",
        value=max_observed,
        unit="m/s^3",
        passed=passed,
        threshold=limit,
        metadata={"violated": not passed},
    )


def curvature_limits_violated(traj: ArrayLike, max_curvature: float) -> MetricResult:
    """Return a MetricResult describing whether curvature exceeds max_curvature."""
    limit = validate_nonnegative(float(max_curvature), name="max_curvature")
    max_observed = float(np.max(curvature(traj)))
    passed = max_observed <= limit
    return MetricResult(
        name="curvature_limits_violated",
        value=max_observed,
        unit="1/m",
        passed=passed,
        threshold=limit,
        metadata={"violated": not passed},
    )


def dynamic_feasibility_score(
    traj: ArrayLike,
    dt: float,
    constraints: Mapping[str, float],
) -> float:
    """Return a 0..1 feasibility score against optional dynamic limits."""
    unknown_constraints = sorted(set(constraints) - _DYNAMIC_FEASIBILITY_CONSTRAINTS)
    if unknown_constraints:
        allowed = ", ".join(sorted(_DYNAMIC_FEASIBILITY_CONSTRAINTS))
        unknown = ", ".join(unknown_constraints)
        raise ValueError(f"unknown dynamic feasibility constraints: {unknown}; allowed: {allowed}")

    if not constraints:
        as_trajectory(traj, name="traj")
        validate_positive(float(dt), name="dt")
        return 1.0

    penalties: list[float] = []
    if "max_speed" in constraints:
        limit = validate_positive(float(constraints["max_speed"]), name="constraints['max_speed']")
        observed = float(np.max(speed_profile(traj, dt)))
        penalties.append(_relative_violation(observed, limit))
    if "max_accel" in constraints:
        limit = validate_positive(float(constraints["max_accel"]), name="constraints['max_accel']")
        observed = float(np.max(vector_norms(acceleration(traj, dt))))
        penalties.append(_relative_violation(observed, limit))
    if "max_jerk" in constraints:
        limit = validate_positive(float(constraints["max_jerk"]), name="constraints['max_jerk']")
        observed = float(np.max(vector_norms(jerk(traj, dt))))
        penalties.append(_relative_violation(observed, limit))
    if "max_curvature" in constraints:
        limit = validate_positive(
            float(constraints["max_curvature"]),
            name="constraints['max_curvature']",
        )
        observed = float(np.max(curvature(traj)))
        penalties.append(_relative_violation(observed, limit))

    if not penalties:
        as_trajectory(traj, name="traj")
        validate_positive(float(dt), name="dt")
        return 1.0
    return float(1.0 / (1.0 + np.mean(penalties)))


def _relative_violation(observed: float, limit: float) -> float:
    return max(0.0, observed / limit - 1.0)

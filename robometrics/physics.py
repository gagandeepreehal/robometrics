"""Physical consistency and dynamic feasibility metrics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robometrics.comfort import acceleration, jerk
from robometrics.geometry import (
    FloatArray,
    as_boolean_mask,
    as_numeric_array,
    as_trajectory,
    require_same_shape,
    validate_nonnegative,
    validate_positive,
    validate_timestamps,
    vector_norms,
)
from robometrics.schemas import MetricResult
from robometrics.trajectory import curvature

_DYNAMIC_FEASIBILITY_CONSTRAINTS = frozenset(
    {"max_speed", "max_accel", "max_jerk", "max_curvature"}
)


def speed_profile(traj: ArrayLike, dt: float) -> FloatArray:
    """Return one speed estimate per trajectory point.

    The returned array has length N for an N-point trajectory. Endpoint speeds
    are finite-difference gradient estimates rather than N-1 interval speeds.
    """
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
    """Return a conservative 0..1 feasibility score against dynamic limits.

    Each configured limit contributes a relative violation
    ``max(0, observed / limit - 1)``. The score is
    ``1 / (1 + worst_violation)`` so adding satisfied constraints cannot hide
    an existing violation. A zero limit is accepted; any positive observation
    against a zero limit produces a score of 0.0.
    """
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
        limit = validate_nonnegative(
            float(constraints["max_speed"]),
            name="constraints['max_speed']",
        )
        observed = float(np.max(speed_profile(traj, dt)))
        penalties.append(_relative_violation(observed, limit))
    if "max_accel" in constraints:
        limit = validate_nonnegative(
            float(constraints["max_accel"]),
            name="constraints['max_accel']",
        )
        observed = float(np.max(vector_norms(acceleration(traj, dt))))
        penalties.append(_relative_violation(observed, limit))
    if "max_jerk" in constraints:
        limit = validate_nonnegative(
            float(constraints["max_jerk"]),
            name="constraints['max_jerk']",
        )
        observed = float(np.max(vector_norms(jerk(traj, dt))))
        penalties.append(_relative_violation(observed, limit))
    if "max_curvature" in constraints:
        limit = validate_nonnegative(
            float(constraints["max_curvature"]),
            name="constraints['max_curvature']",
        )
        observed = float(np.max(curvature(traj)))
        penalties.append(_relative_violation(observed, limit))

    if not penalties:
        as_trajectory(traj, name="traj")
        validate_positive(float(dt), name="dt")
        return 1.0
    return float(1.0 / (1.0 + max(penalties)))


def kinematic_feasibility(
    positions: ArrayLike,
    dt: float = 1.0,
    *,
    timestamps: Optional[ArrayLike] = None,
    max_velocity: Optional[float] = None,
    max_acceleration: Optional[float] = None,
    max_curvature: Optional[float] = None,
) -> float:
    """Return a trajectory kinematic feasibility score.

    Formula:
        Compute finite-difference velocity and acceleration magnitudes from
        position samples shaped ``T`` or ``TxD``. Each configured maximum
        produces a boolean violation mask. The score is
        ``1 - violating_checks / measured_checks`` across all configured masks.

    Inputs:
        Finite positions, either uniform ``dt`` in seconds or strictly
        increasing ``timestamps``, and optional non-negative velocity,
        acceleration, or curvature limits. Curvature requires 2D or 3D
        positions and uses the existing planar XY curvature estimate.

    Output:
        A unitless score in ``[0, 1]`` where higher is more feasible.
        Violations are counted per individual constraint-check slot (e.g. T-1
        velocity checks plus T-2 acceleration checks) rather than per
        timestep with OR across constraints, so a concentrated violation in one
        constraint type is partially offset by passing checks in another.
        Unmeasurable checks on too-short trajectories are skipped; if no
        checks are measurable, the score is ``1.0``.
    """
    position_arr = _as_sample_array(positions, name="positions", allow_scalar=False)
    time_deltas = _time_deltas(position_arr.shape[0], dt=dt, timestamps=timestamps)
    violation_masks: list[NDArray[np.bool_]] = []

    velocities: Optional[FloatArray] = None
    if position_arr.shape[0] >= 2:
        velocities = np.diff(position_arr, axis=0) / time_deltas[:, None]

    if max_velocity is not None and velocities is not None:
        limit = validate_nonnegative(float(max_velocity), name="max_velocity")
        violation_masks.append(vector_norms(velocities) > limit)

    if max_acceleration is not None and velocities is not None and velocities.shape[0] >= 2:
        limit = validate_nonnegative(float(max_acceleration), name="max_acceleration")
        accel_dt = (time_deltas[1:] + time_deltas[:-1]) * 0.5
        accel = np.diff(velocities, axis=0) / accel_dt[:, None]
        violation_masks.append(vector_norms(accel) > limit)

    if max_curvature is not None:
        limit = validate_nonnegative(float(max_curvature), name="max_curvature")
        if position_arr.shape[1] not in (2, 3):
            raise ValueError("max_curvature requires positions with 2 or 3 dimensions")
        violation_masks.append(curvature(position_arr) > limit)

    return _feasibility_score_from_masks(violation_masks)


def dynamic_feasibility(
    mass: float,
    accelerations: ArrayLike,
    *,
    forces: Optional[ArrayLike] = None,
    max_force: Optional[float] = None,
    max_acceleration: Optional[float] = None,
    torques: Optional[ArrayLike] = None,
    max_torque: Optional[float] = None,
    friction_coefficients: Optional[ArrayLike] = None,
    normal_forces: Optional[ArrayLike] = None,
    tangential_forces: Optional[ArrayLike] = None,
) -> float:
    """Return a simple Newtonian dynamic feasibility score.

    Formula:
        Validate mass and acceleration samples, optionally compute required
        force as ``mass * acceleration`` when explicit forces are not provided,
        and evaluate configured max-force, max-acceleration, max-torque, and
        friction-cone checks. The score is
        ``1 - violating_checks / measured_checks``.

    Inputs:
        ``mass`` in kilograms, acceleration vectors in ``m/s^2``, optional
        force/torque samples, and optional non-negative limits. Friction checks
        require coefficient, normal-force, and tangential-force inputs together
        and enforce ``|tangential_force| <= mu * normal_force``.

    Output:
        A unitless score in ``[0, 1]`` where higher is more dynamically
        feasible. If no constraints are configured, valid input returns
        ``1.0``.
    """
    mass_value = validate_positive(float(mass), name="mass")
    accel_arr = _as_sample_array(accelerations, name="accelerations")
    violation_masks: list[NDArray[np.bool_]] = []

    if max_acceleration is not None:
        limit = validate_nonnegative(float(max_acceleration), name="max_acceleration")
        violation_masks.append(vector_norms(accel_arr) > limit)

    force_arr: Optional[FloatArray] = None
    if forces is not None:
        force_arr = _as_sample_array(forces, name="forces")
        require_same_shape(force_arr, accel_arr, "forces", "accelerations")
    elif max_force is not None:
        force_arr = mass_value * accel_arr

    if max_force is not None:
        limit = validate_nonnegative(float(max_force), name="max_force")
        if force_arr is None:
            force_arr = mass_value * accel_arr
        violation_masks.append(vector_norms(force_arr) > limit)

    torque_arr = _as_sample_array(torques, name="torques") if torques is not None else None
    if max_torque is not None:
        if torque_arr is None:
            raise ValueError("torques are required when max_torque is provided")
        limit = validate_nonnegative(float(max_torque), name="max_torque")
        violation_masks.append(vector_norms(torque_arr) > limit)

    friction_args = (friction_coefficients, normal_forces, tangential_forces)
    if any(value is not None for value in friction_args):
        if (
            friction_coefficients is None
            or normal_forces is None
            or tangential_forces is None
        ):
            raise ValueError(
                "friction_coefficients, normal_forces, and tangential_forces "
                "must be provided together"
            )
        violation_masks.append(
            _friction_violation_mask(
                friction_coefficients,
                normal_forces,
                tangential_forces,
            )
        )

    return _feasibility_score_from_masks(violation_masks)


def physics_violation_rate(violations: Union[Mapping[str, ArrayLike], ArrayLike]) -> float:
    """Return fraction of timesteps with any physics violation.

    Formula:
        Boolean or 0/1 violation masks are aggregated with logical OR across
        constraint types, then averaged over timesteps/events.

    Inputs:
        A single mask, a ``CxT`` mask array, or a mapping of same-shaped masks.

    Output:
        A unitless rate in ``[0, 1]`` where lower is better. Empty masks are
        invalid.
    """
    if isinstance(violations, Mapping):
        if not violations:
            raise ValueError("violations must contain at least one mask")
        masks = [
            as_boolean_mask(mask, name=f"violations[{name!r}]")
            for name, mask in violations.items()
        ]
        first_shape = masks[0].shape
        combined = np.zeros(first_shape, dtype=np.bool_)
        for mask in masks:
            if mask.shape != first_shape:
                raise ValueError("all violation masks must have the same shape")
            combined |= mask
    else:
        mask = as_boolean_mask(violations, name="violations")
        combined = np.any(mask, axis=0) if mask.ndim == 2 else mask

    return float(np.mean(combined))


def _relative_violation(observed: float, limit: float) -> float:
    if limit == 0.0:
        return 0.0 if observed == 0.0 else float("inf")
    return max(0.0, observed / limit - 1.0)


def _as_sample_array(
    data: ArrayLike,
    *,
    name: str,
    allow_scalar: bool = True,
) -> FloatArray:
    arr = as_numeric_array(data, name=name)
    if arr.ndim == 0:
        if not allow_scalar:
            raise ValueError(f"{name} must be a T or TxD array")
        return np.asarray(arr.reshape(1, 1), dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one sample")
        return np.asarray(arr[:, None], dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError(f"{name} must have non-empty sample and feature dimensions")
        return arr
    raise ValueError(f"{name} must be a T or TxD array")


def _time_deltas(
    timesteps: int,
    *,
    dt: float,
    timestamps: Optional[ArrayLike],
) -> FloatArray:
    if timestamps is not None:
        timestamp_arr = validate_timestamps(timestamps)
        if timestamp_arr.shape[0] != timesteps:
            raise ValueError("timestamps must have one value per position sample")
        return np.asarray(np.diff(timestamp_arr), dtype=np.float64)

    timestep = validate_positive(float(dt), name="dt")
    return np.full(max(timesteps - 1, 0), timestep, dtype=np.float64)


def _feasibility_score_from_masks(masks: list[NDArray[np.bool_]]) -> float:
    total = 0
    violation_count = 0
    for mask in masks:
        total += int(mask.size)
        violation_count += int(np.count_nonzero(mask))
    if total == 0:
        return 1.0
    return float(1.0 - violation_count / total)


def _as_1d_numeric(data: ArrayLike, *, name: str) -> FloatArray:
    arr = as_numeric_array(data, name=name)
    if arr.ndim == 0:
        return np.asarray(arr.reshape(1), dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array or scalar")
    return arr


def _friction_violation_mask(
    friction_coefficients: ArrayLike,
    normal_forces: ArrayLike,
    tangential_forces: ArrayLike,
) -> NDArray[np.bool_]:
    normal = _as_1d_numeric(normal_forces, name="normal_forces")
    if np.any(normal < 0.0):
        raise ValueError("normal_forces must be non-negative")

    tangential = as_numeric_array(tangential_forces, name="tangential_forces")
    if tangential.ndim == 1:
        tangential_magnitude = np.abs(tangential)
    elif tangential.ndim == 2:
        tangential_magnitude = vector_norms(tangential)
    else:
        raise ValueError("tangential_forces must be a 1D or 2D array")
    if tangential_magnitude.shape != normal.shape:
        raise ValueError("normal_forces and tangential_forces must have matching samples")

    mu_raw = as_numeric_array(friction_coefficients, name="friction_coefficients")
    if mu_raw.ndim == 0:
        mu = np.full(normal.shape, float(mu_raw.item()), dtype=np.float64)
    else:
        mu = _as_1d_numeric(mu_raw, name="friction_coefficients")
        if mu.shape != normal.shape:
            raise ValueError("friction_coefficients must be scalar or match normal_forces")
    if np.any(mu < 0.0):
        raise ValueError("friction_coefficients must be non-negative")

    return np.asarray(tangential_magnitude > mu * normal, dtype=np.bool_)

"""Manipulation metric pack."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from robometrics.geometry import (
    FloatArray,
    as_boolean_mask,
    as_trajectory,
    require_same_shape,
    validate_positive,
)
from robometrics.trajectory import average_displacement_error


def grasp_success_rate(
    attempts: ArrayLike,
    successes: ArrayLike,
) -> float:
    """Return fraction of grasp attempts that succeeded.

    Formula: count(attempts & successes) / count(attempts)

    Reference: Mahler et al., Dex-Net 2.0: Deep Learning to Plan Robust
               Grasps with Synthetic Point Clouds, RSS 2017.

    Inputs:
        attempts: boolean or 0/1 1D array, True where a grasp was attempted.
        successes: boolean or 0/1 1D array with the same shape, True where
                   the grasp succeeded. Success values outside attempt
                   timesteps are ignored.

    Output:
        A rate in [0, 1] where 1.0 is perfect. Returns nan if no attempts.
    """
    attempt_mask = as_boolean_mask(attempts, name="attempts")
    success_mask = as_boolean_mask(successes, name="successes")
    if attempt_mask.shape != success_mask.shape:
        raise ValueError("attempts and successes must have the same shape")
    attempt_count = int(np.sum(attempt_mask))
    if attempt_count == 0:
        return float("nan")
    return float(np.sum(attempt_mask & success_mask) / attempt_count)


def contact_richness(
    contact_forces: ArrayLike,
    threshold: float = 0.1,
) -> float:
    """Return fraction of timesteps with meaningful contact force magnitude.

    Formula: mean(||contact_forces[t]|| > threshold)

    Reference: Handa et al., DexPilot: Vision-Based Teleoperation of
               Dexterous Robotic Hand-Arm System, ICRA 2020.

    Inputs:
        contact_forces: TxD array of contact force vectors or T-length scalar
                        force magnitudes. D is typically 3 for XYZ forces.
        threshold: positive scalar force magnitude threshold in Newtons.

    Output:
        A unitless score in [0, 1] where higher means more contact engagement.
    """
    magnitudes = _force_magnitudes(contact_forces, name="contact_forces")
    threshold_value = validate_positive(float(threshold), name="threshold")
    return float(np.mean(magnitudes > threshold_value))


def force_limit_compliance(
    forces: ArrayLike,
    max_force: float,
) -> float:
    """Return fraction of timesteps where force magnitude is within limit.

    Formula: mean(||forces[t]|| <= max_force)

    Reference: Standard safety limit used in ISO/TS 15066 collaborative
               robot safety standard.

    Inputs:
        forces: TxD or T-length force array in Newtons.
        max_force: positive scalar maximum allowable force in Newtons.

    Output:
        A score in [0, 1] where 1.0 means always within limit.
    """
    magnitudes = _force_magnitudes(forces, name="forces")
    max_force_value = validate_positive(float(max_force), name="max_force")
    return float(np.mean(magnitudes <= max_force_value))


def joint_limit_violation_rate(
    joint_angles: ArrayLike,
    lower_limits: ArrayLike,
    upper_limits: ArrayLike,
) -> float:
    """Return fraction of configs where any joint exceeds its limits.

    Formula:
        For each timestep t, check if any joint j satisfies
        joint_angles[t, j] < lower_limits[j] or > upper_limits[j].
        Return mean(any_violated_mask).

    Reference: Standard kinematic constraint used in motion planning;
               see Siciliano et al., Robotics: Modelling, Planning and
               Control, Springer 2009.

    Inputs:
        joint_angles: TxJ array of joint angles in radians (or any unit
                      consistent with the limits).
        lower_limits: J-length array of lower joint limits.
        upper_limits: J-length array of upper joint limits.

    Output:
        A rate in [0, 1] where 0.0 means no violations.
    """
    angles = np.asarray(joint_angles, dtype=np.float64)
    if angles.ndim != 2:
        raise ValueError("joint_angles must be a TxJ array")
    if angles.shape[0] == 0 or angles.shape[1] == 0:
        raise ValueError("joint_angles must contain at least one config and one joint")
    if not np.all(np.isfinite(angles)):
        raise ValueError("joint_angles must contain only finite values")

    lower = _joint_limits(lower_limits, name="lower_limits")
    upper = _joint_limits(upper_limits, name="upper_limits")
    if lower.shape[0] != angles.shape[1] or upper.shape[0] != angles.shape[1]:
        raise ValueError("lower_limits and upper_limits must match joint_angles width")
    if not np.all(upper > lower):
        raise ValueError("upper_limits must be greater than lower_limits")

    violated = np.any((angles < lower[None, :]) | (angles > upper[None, :]), axis=1)
    return float(np.mean(violated))


def end_effector_tracking_error(
    ee_traj: ArrayLike,
    target_traj: ArrayLike,
) -> float:
    """Return mean Euclidean distance between end-effector and target path.

    Formula: mean over t of ||ee_traj[t] - target_traj[t]||

    Reference: Standard Cartesian task-space error; see Siciliano et al.,
               Robotics: Modelling, Planning and Control, Springer 2009.

    Inputs:
        ee_traj: Nx2 or Nx3 array of end-effector positions.
        target_traj: Nx2 or Nx3 array of target positions (same shape).

    Output:
        Mean tracking error in position units. 0.0 means perfect tracking.
    """
    ee = as_trajectory(ee_traj, name="ee_traj")
    target = as_trajectory(target_traj, name="target_traj")
    require_same_shape(ee, target, "ee_traj", "target_traj")
    return average_displacement_error(ee, target)


def _force_magnitudes(data: ArrayLike, *, name: str) -> FloatArray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim not in (1, 2):
        raise ValueError(f"{name} must be a T-length or TxD array")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one timestep")
    if arr.ndim == 2 and arr.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one force dimension")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if arr.ndim == 1:
        return np.asarray(np.abs(arr), dtype=np.float64)
    return np.asarray(np.linalg.norm(arr, axis=1), dtype=np.float64)


def _joint_limits(data: ArrayLike, *, name: str) -> FloatArray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a J-length array")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one joint limit")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr

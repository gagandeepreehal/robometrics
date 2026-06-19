"""Task-level robotics metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from robometrics.geometry import as_1d_array, as_points, require_same_shape, validate_positive


def task_success_rate(outcomes: ArrayLike) -> float:
    """Return fraction of task attempts that succeeded.

    Formula: mean(bool(outcomes))
    Reference: Standard binary task evaluation used in robotics benchmarks.

    Inputs: Boolean or 0/1 1D array of per-episode outcomes.
    Output: A rate in [0, 1] where 1.0 is perfect success.
    """
    outcome_arr = as_1d_array(outcomes, name="outcomes")
    if not np.all((outcome_arr == 0.0) | (outcome_arr == 1.0)):
        raise ValueError("outcomes must contain only boolean or 0/1 values")
    return float(np.mean(outcome_arr.astype(np.bool_)))


def goal_reaching_accuracy(
    positions: ArrayLike,
    goals: ArrayLike,
    tolerance: float,
) -> float:
    """Return fraction of goals reached within Euclidean tolerance.

    Formula: mean(||positions[i] - goals[i]|| <= tolerance)
    Reference: Standard navigation evaluation; see Anderson et al.,
               Habitat: A Platform for Embodied AI Research, ICCV 2019.

    Inputs:
        positions: NxD array of final agent positions.
        goals: NxD array of goal positions (same shape as positions).
        tolerance: positive scalar distance threshold in position units.
    Output: A rate in [0, 1] where 1.0 means all goals reached.
    """
    position_arr = as_points(positions, name="positions")
    goal_arr = as_points(goals, name="goals")
    require_same_shape(position_arr, goal_arr, "positions", "goals")
    tolerance_value = validate_positive(float(tolerance), name="tolerance")
    distances = np.linalg.norm(position_arr - goal_arr, axis=1)
    return float(np.mean(distances <= tolerance_value))

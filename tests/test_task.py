from __future__ import annotations

import numpy as np
import pytest

from robometrics import goal_reaching_accuracy, task_success_rate


def test_task_success_rate_all_successes_and_failures() -> None:
    assert task_success_rate([True, True, True]) == 1.0
    assert task_success_rate([0, 0, 0]) == 0.0


def test_goal_reaching_accuracy_all_goals_within_tolerance() -> None:
    positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    goals = np.array([[0.2, 0.0], [1.0, 1.1], [2.3, 2.0]])

    assert goal_reaching_accuracy(positions, goals, tolerance=0.5) == 1.0


def test_goal_reaching_accuracy_no_goals_within_tolerance() -> None:
    positions = np.array([[0.0, 0.0], [1.0, 1.0]])
    goals = np.array([[2.0, 0.0], [1.0, 3.0]])

    assert goal_reaching_accuracy(positions, goals, tolerance=0.5) == 0.0


def test_task_metrics_reject_empty_arrays() -> None:
    with pytest.raises(ValueError, match="outcomes must contain at least one value"):
        task_success_rate([])

    with pytest.raises(ValueError, match="positions must contain at least one point"):
        goal_reaching_accuracy(np.empty((0, 2)), np.empty((0, 2)), tolerance=1.0)


def test_goal_reaching_accuracy_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="positions and goals must have the same shape"):
        goal_reaching_accuracy([[0.0, 0.0]], [[0.0, 0.0], [1.0, 1.0]], tolerance=1.0)


def test_task_metrics_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="outcomes must contain only boolean or 0/1 values"):
        task_success_rate([0, 1, 2])

    with pytest.raises(ValueError, match="tolerance must be a positive finite value"):
        goal_reaching_accuracy([[0.0, 0.0]], [[0.0, 0.0]], tolerance=0.0)

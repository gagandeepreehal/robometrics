from __future__ import annotations

import numpy as np

from robometrics import (
    acceleration_limits_violated,
    curvature_limits_violated,
    dynamic_feasibility_score,
    jerk_limits_violated,
    speed_profile,
)


def test_speed_profile_for_constant_velocity() -> None:
    traj = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    assert np.allclose(speed_profile(traj, dt=1.0), np.ones(3))


def test_acceleration_limit_result() -> None:
    t = np.arange(5, dtype=np.float64)
    traj = np.column_stack((t**2, np.zeros_like(t)))

    result = acceleration_limits_violated(traj, dt=1.0, max_accel=1.0)

    assert result.passed is False
    assert result.metadata["violated"] is True
    assert result.value > result.threshold


def test_jerk_and_curvature_limit_results() -> None:
    t = np.arange(6, dtype=np.float64)
    jerky = np.column_stack((t**3, np.zeros_like(t)))
    turn = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])

    assert jerk_limits_violated(jerky, dt=1.0, max_jerk=1.0).passed is False
    assert curvature_limits_violated(turn, max_curvature=0.1).passed is False


def test_dynamic_feasibility_score() -> None:
    smooth = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    aggressive = np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0], [9.0, 0.0]])

    assert dynamic_feasibility_score(smooth, 1.0, {"max_accel": 1.0}) == 1.0
    assert dynamic_feasibility_score(aggressive, 1.0, {"max_accel": 1.0}) < 1.0
    assert dynamic_feasibility_score(smooth, 1.0, {}) == 1.0

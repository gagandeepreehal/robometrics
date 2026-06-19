from __future__ import annotations

import numpy as np
import pytest

from robometrics import (
    acceleration,
    jerk,
    jerk_cost,
    max_acceleration,
    max_deceleration,
    smoothness_score,
)


def test_constant_velocity_has_zero_acceleration_and_jerk() -> None:
    traj = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

    assert np.allclose(acceleration(traj, dt=1.0), np.zeros_like(traj))
    assert np.allclose(jerk(traj, dt=1.0), np.zeros_like(traj))
    assert jerk_cost(traj, dt=1.0) == 0.0
    assert smoothness_score(traj, dt=1.0) == 1.0


def test_smoothness_score_does_not_change_with_dt_alone() -> None:
    traj = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.01],
            [2.0, -0.01],
            [3.0, 0.02],
            [4.0, -0.02],
            [5.0, 0.0],
        ]
    )

    assert smoothness_score(traj, dt=0.01) == pytest.approx(smoothness_score(traj, dt=1.0))


def test_smoothness_score_short_trajectories_are_degenerate() -> None:
    traj = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    assert smoothness_score(traj, dt=0.1) == 1.0


def test_smoothness_score_distinguishes_bad_from_catastrophic() -> None:
    traj = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ]
    )

    bad = smoothness_score(traj * 10.0, dt=0.1)
    catastrophic = smoothness_score(traj * 100.0, dt=0.1)

    assert catastrophic < bad
    assert bad - catastrophic > 0.01


def test_quadratic_motion_has_constant_acceleration() -> None:
    t = np.arange(5, dtype=np.float64)
    traj = np.column_stack((t**2, np.zeros_like(t)))

    assert max_acceleration(traj, dt=1.0) == pytest.approx(2.0)
    assert jerk_cost(traj, dt=1.0) == pytest.approx(0.0)


def test_max_deceleration_for_slowing_trajectory() -> None:
    traj = np.array([[0.0, 0.0], [3.0, 0.0], [5.0, 0.0], [6.0, 0.0], [6.0, 0.0]])

    assert max_deceleration(traj, dt=1.0) > 0.0


def test_stationary_and_single_point_trajectories_are_well_defined() -> None:
    stationary = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    single = np.array([[1.0, 1.0]])

    assert max_acceleration(stationary, dt=0.1) == 0.0
    assert np.allclose(acceleration(single, dt=0.1), np.zeros((1, 2)))


def test_invalid_dt_raises() -> None:
    with pytest.raises(ValueError):
        acceleration(np.array([[0.0, 0.0], [1.0, 0.0]]), dt=0.0)

from __future__ import annotations

import numpy as np
import pytest

from robometrics import (
    acceleration,
    acceleration_magnitude,
    jerk,
    jerk_cost,
    jerk_magnitude,
    max_acceleration,
    max_deceleration,
    mean_acceleration,
    rms_acceleration,
    smoothness_score,
)


def test_constant_velocity_has_zero_acceleration_and_jerk() -> None:
    traj = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

    assert np.allclose(acceleration(traj, dt=1.0), np.zeros_like(traj))
    assert np.allclose(jerk(traj, dt=1.0), np.zeros_like(traj))
    assert jerk_cost(traj, dt=1.0) == 0.0
    assert smoothness_score(traj) == 1.0


def test_smoothness_score_is_spatial_scale_invariant() -> None:
    theta = np.linspace(0.0, np.pi, 32)
    meters = np.column_stack((5.0 * np.cos(theta), 5.0 * np.sin(theta)))
    centimeters = meters * 100.0

    assert smoothness_score(centimeters) == pytest.approx(smoothness_score(meters))


def test_smoothness_score_short_trajectories_are_degenerate() -> None:
    traj = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    with pytest.warns(RuntimeWarning, match="fewer than four points"):
        assert smoothness_score(traj) == 1.0


def test_smoothness_score_penalizes_rough_shape() -> None:
    smooth = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
        ]
    )
    jerky = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, -1.0],
            [3.0, 1.0],
            [4.0, -1.0],
            [5.0, 1.0],
        ]
    )

    smooth_score = smoothness_score(smooth)
    jerky_score = smoothness_score(jerky)

    assert jerky_score < smooth_score
    assert smooth_score - jerky_score > 0.1


def test_quadratic_motion_has_constant_acceleration() -> None:
    t = np.arange(5, dtype=np.float64)
    traj = np.column_stack((t**2, np.zeros_like(t)))

    assert np.allclose(acceleration_magnitude(traj, dt=1.0), np.full(5, 2.0))
    assert max_acceleration(traj, dt=1.0) == pytest.approx(2.0)
    assert mean_acceleration(traj, dt=1.0) == pytest.approx(2.0)
    assert rms_acceleration(traj, dt=1.0) == pytest.approx(2.0)
    assert np.allclose(jerk_magnitude(traj, dt=1.0), np.zeros(5))
    assert jerk_cost(traj, dt=1.0) == pytest.approx(0.0)


def test_quadratic_motion_jerk_clips_roundoff_noise() -> None:
    t = np.arange(6, dtype=np.float64)
    traj = np.column_stack((t**2, np.zeros_like(t)))

    assert np.array_equal(jerk(traj, dt=0.5), np.zeros_like(traj))


def test_max_deceleration_for_slowing_trajectory() -> None:
    traj = np.array([[0.0, 0.0], [3.0, 0.0], [5.0, 0.0], [6.0, 0.0], [6.0, 0.0]])

    assert max_deceleration(traj, dt=1.0) > 0.0


def test_stationary_and_single_point_trajectories_are_well_defined() -> None:
    stationary = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    single = np.array([[1.0, 1.0]])
    repeated = np.zeros((4, 2), dtype=np.float64)

    assert max_acceleration(stationary, dt=0.1) == 0.0
    assert max_deceleration(single, dt=0.1) == 0.0
    assert np.allclose(acceleration(single, dt=0.1), np.zeros((1, 2)))
    assert smoothness_score(repeated) == 1.0


def test_invalid_dt_raises() -> None:
    with pytest.raises(ValueError):
        acceleration(np.array([[0.0, 0.0], [1.0, 0.0]]), dt=0.0)

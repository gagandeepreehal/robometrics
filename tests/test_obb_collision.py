from __future__ import annotations

import numpy as np
import pytest

from robometrics import collision_rate_obb
from robometrics.geometry import obb_overlap


def test_collision_rate_obb_identical_axis_aligned_boxes_collide() -> None:
    rate = collision_rate_obb(
        ego_traj=np.array([[0.0, 0.0]]),
        ego_dims=np.array([4.0, 2.0]),
        ego_yaws=np.array([0.0]),
        actor_trajs=[np.array([[0.0, 0.0]])],
        actor_dims=[np.array([4.0, 2.0])],
        actor_yaws=[np.array([0.0])],
    )

    assert rate == 1.0


def test_collision_rate_obb_far_boxes_do_not_collide() -> None:
    rate = collision_rate_obb(
        ego_traj=np.array([[0.0, 0.0]]),
        ego_dims=np.array([4.0, 2.0]),
        ego_yaws=np.array([0.0]),
        actor_trajs=[np.array([[100.0, 0.0]])],
        actor_dims=[np.array([4.0, 2.0])],
        actor_yaws=[np.array([0.0])],
    )

    assert rate == 0.0


def test_collision_rate_obb_thin_boxes_do_not_collide_when_disc_approximation_would() -> None:
    rate = collision_rate_obb(
        ego_traj=np.array([[0.0, 0.0]]),
        ego_dims=np.array([10.0, 0.2]),
        ego_yaws=np.array([0.0]),
        actor_trajs=[np.array([[0.0, 0.6]])],
        actor_dims=[np.array([10.0, 0.2])],
        actor_yaws=[np.array([0.05])],
    )

    assert rate == 0.0


def test_collision_rate_obb_long_perpendicular_boxes_collide_at_corner() -> None:
    rate = collision_rate_obb(
        ego_traj=np.array([[0.0, 0.0]]),
        ego_dims=np.array([10.0, 0.2]),
        ego_yaws=np.array([0.0]),
        actor_trajs=[np.array([[4.8, 4.8]])],
        actor_dims=[np.array([10.0, 0.2])],
        actor_yaws=[np.array([np.pi / 2.0])],
    )

    assert rate > 0.0


def test_collision_rate_obb_rejects_bad_ego_shape() -> None:
    with pytest.raises(ValueError, match="ego_traj must be an Nx2 array"):
        collision_rate_obb(
            ego_traj=np.array([0.0, 0.0]),
            ego_dims=np.array([4.0, 2.0]),
            ego_yaws=np.array([0.0]),
            actor_trajs=[],
            actor_dims=[],
            actor_yaws=[],
        )


def test_collision_rate_obb_rejects_actor_yaw_length_mismatch() -> None:
    with pytest.raises(ValueError, match="actor_yaws\\[0\\]"):
        collision_rate_obb(
            ego_traj=np.array([[0.0, 0.0], [1.0, 0.0]]),
            ego_dims=np.array([4.0, 2.0]),
            ego_yaws=np.array([0.0, 0.0]),
            actor_trajs=[np.array([[0.0, 0.0], [1.0, 0.0]])],
            actor_dims=[np.array([4.0, 2.0])],
            actor_yaws=[np.array([0.0])],
        )


def test_collision_rate_obb_rejects_actor_dims_count_mismatch() -> None:
    with pytest.raises(ValueError, match="got 0 dims entries for 1 actor_trajs"):
        collision_rate_obb(
            ego_traj=np.array([[0.0, 0.0]]),
            ego_dims=np.array([4.0, 2.0]),
            ego_yaws=np.array([0.0]),
            actor_trajs=[np.array([[0.0, 0.0]])],
            actor_dims=[],
            actor_yaws=[np.array([0.0])],
        )


def test_obb_overlap_axis_aligned_helper() -> None:
    assert obb_overlap(
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
        0.0,
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
        0.0,
    )
    assert not obb_overlap(
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
        0.0,
        np.array([3.0, 0.0]),
        np.array([1.0, 1.0]),
        0.0,
    )

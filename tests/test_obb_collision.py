from __future__ import annotations

import numpy as np
import pytest

from robometrics import collision_rate_obb
from robometrics.geometry import obb_overlap, obb_overlap_batch


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


def test_obb_overlap_batch_matches_scalar_helper() -> None:
    centers_a = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    half_extents_a = np.array([[1.0, 1.0], [2.0, 0.5], [1.0, 0.25]])
    yaws_a = np.array([0.0, 0.25, np.pi / 4.0])
    centers_b = np.array([[0.5, 0.0], [10.0, 0.0], [1.2, 1.2]])
    half_extents_b = np.array([[1.0, 1.0], [2.0, 0.5], [0.5, 0.5]])
    yaws_b = np.array([0.0, -0.25, -np.pi / 6.0])

    batch = obb_overlap_batch(
        centers_a,
        half_extents_a,
        yaws_a,
        centers_b,
        half_extents_b,
        yaws_b,
    )
    scalar = np.array(
        [
            obb_overlap(ca, ea, float(ya), cb, eb, float(yb))
            for ca, ea, ya, cb, eb, yb in zip(
                centers_a,
                half_extents_a,
                yaws_a,
                centers_b,
                half_extents_b,
                yaws_b,
            )
        ]
    )

    assert np.array_equal(batch, scalar)


def test_obb_overlap_batch_validates_shapes_and_values() -> None:
    valid_centers = np.zeros((1, 2))
    valid_extents = np.ones((1, 2))
    valid_yaws = np.zeros(1)

    with pytest.raises(ValueError, match="same number of rows"):
        obb_overlap_batch(
            np.zeros((2, 2)),
            np.ones((2, 2)),
            np.zeros(2),
            valid_centers,
            valid_extents,
            valid_yaws,
        )
    with pytest.raises(ValueError, match="half_extents_a"):
        obb_overlap_batch(
            valid_centers,
            -valid_extents,
            valid_yaws,
            valid_centers,
            valid_extents,
            valid_yaws,
        )
    with pytest.raises(ValueError, match="centers_a"):
        obb_overlap_batch(
            np.zeros(2),
            valid_extents,
            valid_yaws,
            valid_centers,
            valid_extents,
            valid_yaws,
        )
    with pytest.raises(ValueError, match="yaws_a"):
        obb_overlap_batch(
            valid_centers,
            valid_extents,
            np.zeros((1, 1)),
            valid_centers,
            valid_extents,
            valid_yaws,
        )

    empty = obb_overlap_batch(
        np.zeros((0, 2)),
        np.zeros((0, 2)),
        np.zeros(0),
        np.zeros((0, 2)),
        np.zeros((0, 2)),
        np.zeros(0),
    )
    assert empty.shape == (0,)

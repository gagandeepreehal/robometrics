from __future__ import annotations

from math import inf

import numpy as np
import pytest

from robometrics import (
    collision_rate,
    lane_departure_rate,
    min_distance_to_actors,
    time_to_collision,
)
from robometrics.schemas import AgentState


def test_collision_rate_counts_time_aligned_collisions() -> None:
    ego = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    actors = np.array([[[10.0, 0.0], [1.4, 0.0], [10.0, 0.0]]])

    assert collision_rate(ego, actors, ego_radius=0.5, actor_radius=0.5) == pytest.approx(1.0 / 3.0)


def test_collision_rate_no_actors_or_no_collision() -> None:
    ego = np.array([[0.0, 0.0], [1.0, 0.0]])
    far_actor = np.array([[[10.0, 0.0], [11.0, 0.0]]])

    assert collision_rate(ego, [], ego_radius=1.0, actor_radius=1.0) == 0.0
    assert collision_rate(ego, far_actor, ego_radius=0.5, actor_radius=0.5) == 0.0


def test_time_to_collision_constant_velocity() -> None:
    ego = AgentState(x=0.0, y=0.0, vx=1.0, vy=0.0, radius=1.0)
    actor = AgentState(x=10.0, y=0.0, vx=0.0, vy=0.0, radius=1.0)

    assert time_to_collision(ego, actor) == pytest.approx(8.0)
    assert time_to_collision([0.0, 0.0, -1.0, 0.0], [10.0, 0.0, 0.0, 0.0]) == inf
    assert time_to_collision([0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 1.0]) == 0.0


def test_min_distance_to_actors() -> None:
    ego = np.array([[0.0, 0.0], [1.0, 0.0]])
    actors = [np.array([[5.0, 0.0], [1.5, 0.0]])]

    assert min_distance_to_actors(ego, actors) == pytest.approx(0.5)
    assert min_distance_to_actors(ego, []) == inf


def test_lane_departure_rate_uses_polygon_boundary() -> None:
    ego = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    lane = np.array([[-1.0, -1.0], [2.0, -1.0], [2.0, 1.0], [-1.0, 1.0]])

    assert lane_departure_rate(ego, lane) == pytest.approx(1.0 / 3.0)


def test_lane_departure_accepts_closed_polygon() -> None:
    ego = np.array([[0.0, 0.0], [3.0, 0.0]])
    lane = np.array(
        [[-1.0, -1.0], [2.0, -1.0], [2.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]]
    )

    assert lane_departure_rate(ego, lane) == pytest.approx(0.5)


def test_safety_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        collision_rate(np.array([[0.0, 0.0]]), [], ego_radius=-1.0, actor_radius=1.0)
    with pytest.raises(ValueError):
        lane_departure_rate(np.array([[0.0, 0.0]]), np.array([[0.0, 0.0], [1.0, 0.0]]))


def test_safety_rejects_bad_actor_trajectory_shape() -> None:
    ego = np.array([[0.0, 0.0], [1.0, 0.0]])

    with pytest.raises(ValueError, match="actor_trajs"):
        collision_rate(
            ego,
            [np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([0.0, 0.0])],
            ego_radius=0.5,
            actor_radius=0.5,
        )

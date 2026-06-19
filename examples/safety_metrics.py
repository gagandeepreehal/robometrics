from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robometrics import (
    AgentState,
    collision_rate,
    collision_rate_obb,
    min_distance_to_actors,
    time_to_collision,
)


def main() -> None:
    ego = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    actors = [
        np.array([[0.0, 2.0], [1.0, 1.5], [2.0, 0.8], [3.0, 0.4]]),
        np.array([[0.0, -3.0], [1.0, -3.0], [2.0, -3.0], [3.0, -3.0]]),
    ]

    print(f"Minimum actor distance: {min_distance_to_actors(ego, actors):.3f} m")
    print(f"Collision rate: {collision_rate(ego, actors, ego_radius=0.3, actor_radius=0.3):.3f}")

    ego_state = AgentState(x=0.0, y=0.0, vx=2.0, vy=0.0, radius=0.3)
    actor_state = AgentState(x=10.0, y=0.0, vx=0.0, vy=0.0, radius=0.3)
    print(f"Time to collision: {time_to_collision(ego_state, actor_state):.3f} s")

    ego_dims = np.array([4.5, 2.0])
    ego_yaws = np.zeros(ego.shape[0])
    actor_dims = [np.array([4.5, 2.0]), np.array([4.5, 2.0])]
    actor_yaws = [np.zeros(actor.shape[0]) for actor in actors]
    obb_rate = collision_rate_obb(ego, ego_dims, ego_yaws, actors, actor_dims, actor_yaws)
    print(f"OBB collision rate: {obb_rate:.3f}")


if __name__ == "__main__":
    main()

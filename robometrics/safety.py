"""Geometry-based safety metrics."""

from __future__ import annotations

from math import inf, sqrt
from typing import Any, Union

import numpy as np
from numpy.typing import ArrayLike

from robometrics.geometry import (
    as_actor_trajectories,
    as_trajectory,
    points_in_polygon,
    validate_nonnegative,
    xy,
)
from robometrics.schemas import AgentState


def collision_rate(
    ego_traj: ArrayLike,
    actor_trajs: object,
    ego_radius: float,
    actor_radius: float,
) -> float:
    """Return fraction of actor-covered ego timesteps that collide with at least one actor."""
    ego = as_trajectory(ego_traj, name="ego_traj")
    actors = as_actor_trajectories(actor_trajs)
    ego_r = validate_nonnegative(float(ego_radius), name="ego_radius")
    actor_r = validate_nonnegative(float(actor_radius), name="actor_radius")
    if not actors:
        return 0.0

    covered_steps = np.zeros(ego.shape[0], dtype=np.bool_)
    collision_steps = np.zeros(ego.shape[0], dtype=np.bool_)
    threshold = ego_r + actor_r
    for actor in actors:
        overlap = min(ego.shape[0], actor.shape[0])
        if overlap == 0:
            continue
        covered_steps[:overlap] = True
        distances = np.linalg.norm(xy(ego[:overlap]) - xy(actor[:overlap]), axis=1)
        collision_steps[:overlap] |= distances <= threshold
    if not np.any(covered_steps):
        return 0.0
    return float(np.mean(collision_steps[covered_steps]))


def time_to_collision(
    ego_state: Union[AgentState, ArrayLike, dict[str, Any]],
    actor_state: Union[AgentState, ArrayLike, dict[str, Any]],
) -> float:
    """Return constant-velocity time to collision for two disc agents.

    States must provide x, y, vx, and vy. Radius is optional and defaults to 0.
    A non-colliding or diverging pair returns math.inf.
    """
    ego = _coerce_agent_state(ego_state)
    actor = _coerce_agent_state(actor_state)

    relative_position = np.array([actor.x - ego.x, actor.y - ego.y], dtype=np.float64)
    relative_velocity = np.array([actor.vx - ego.vx, actor.vy - ego.vy], dtype=np.float64)
    radius = ego.radius + actor.radius

    c = float(np.dot(relative_position, relative_position) - radius * radius)
    if c <= 0.0:
        return 0.0

    a = float(np.dot(relative_velocity, relative_velocity))
    if a <= 1e-12:
        return inf

    b = 2.0 * float(np.dot(relative_position, relative_velocity))
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return inf

    root = sqrt(discriminant)
    candidates = [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]
    future = [value for value in candidates if value >= 0.0]
    return float(min(future)) if future else inf


def min_distance_to_actors(ego_traj: ArrayLike, actor_trajs: object) -> float:
    """Return minimum time-aligned XY distance from ego to any actor."""
    ego = as_trajectory(ego_traj, name="ego_traj")
    actors = as_actor_trajectories(actor_trajs)
    if not actors:
        return inf

    min_distance = inf
    for actor in actors:
        overlap = min(ego.shape[0], actor.shape[0])
        if overlap == 0:
            continue
        distances = np.linalg.norm(xy(ego[:overlap]) - xy(actor[:overlap]), axis=1)
        min_distance = min(min_distance, float(np.min(distances)))
    return min_distance


def lane_departure_rate(ego_traj: ArrayLike, lane_boundary: ArrayLike) -> float:
    """Return fraction of ego points outside a polygonal lane boundary."""
    ego = as_trajectory(ego_traj, name="ego_traj")
    boundary = as_trajectory(lane_boundary, name="lane_boundary")
    inside = points_in_polygon(ego, boundary)
    return float(np.mean(~inside))


def _coerce_agent_state(state: Union[AgentState, ArrayLike, dict[str, Any]]) -> AgentState:
    if isinstance(state, AgentState):
        return state
    if isinstance(state, dict):
        return AgentState(**state)

    arr = np.asarray(state, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] < 4:
        raise ValueError("agent state arrays must contain at least [x, y, vx, vy]")
    radius = float(arr[4]) if arr.shape[0] > 4 else 0.0
    return AgentState(
        x=float(arr[0]),
        y=float(arr[1]),
        vx=float(arr[2]),
        vy=float(arr[3]),
        radius=radius,
    )

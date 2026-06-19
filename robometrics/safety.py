"""Geometry-based safety metrics."""

from __future__ import annotations

from math import inf, sqrt
from typing import Any, Union

import numpy as np
from numpy.typing import ArrayLike

from robometrics.geometry import (
    FloatArray,
    as_actor_trajectories,
    as_trajectory,
    obb_overlap,
    points_in_polygon,
    validate_nonnegative,
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
        distances = np.linalg.norm(
            _positions(ego[:overlap]) - _positions(actor[:overlap]),
            axis=1,
        )
        collision_steps[:overlap] |= distances <= threshold
    if not np.any(covered_steps):
        return 0.0
    return float(np.mean(collision_steps[covered_steps]))


def collision_rate_obb(
    ego_traj: ArrayLike,
    ego_dims: ArrayLike,
    ego_yaws: ArrayLike,
    actor_trajs: object,
    actor_dims: ArrayLike,
    actor_yaws: object,
) -> float:
    """Return fraction of actor-covered ego timesteps with OBB collision.

    Formula:
        For each timestep t covered by at least one actor, check if ego OBB
        overlaps with any actor OBB using the Separating Axis Theorem.
        Return mean(collision_mask[covered_steps]).

    Reference: Standard OBB collision check; Gottschalk et al.,
               OBBTree, SIGGRAPH 1996.

    Inputs:
        ego_traj: Nx2 ego center positions (XY only).
        ego_dims: (2,) array [length, width] in meters, or Nx2 per-timestep.
        ego_yaws: N-length array of ego heading angles in radians.
        actor_trajs: list of Mx2 actor center position arrays.
        actor_dims: list of (2,) or Mx2 arrays per actor, [length, width].
        actor_yaws: list of M-length yaw arrays, one per actor.

    Output:
        A rate in [0, 1] where 0.0 means no OBB collisions.
    """
    ego = _as_xy_trajectory(ego_traj, name="ego_traj", allow_empty=False)
    ego_yaw_values = _as_yaw_array(ego_yaws, name="ego_yaws", length=ego.shape[0])
    ego_dim_values = _as_dims_array(ego_dims, name="ego_dims", length=ego.shape[0])
    actors = _as_actor_xy_list(actor_trajs)
    actor_dim_values = _as_actor_dims_list(actor_dims, actors)
    actor_yaw_values = _as_actor_yaws_list(actor_yaws, actors)
    if not actors:
        return 0.0

    covered_steps = np.zeros(ego.shape[0], dtype=np.bool_)
    collision_steps = np.zeros(ego.shape[0], dtype=np.bool_)
    for actor_index, actor in enumerate(actors):
        overlap = min(ego.shape[0], actor.shape[0])
        if overlap == 0:
            continue
        covered_steps[:overlap] = True
        dims = actor_dim_values[actor_index]
        yaws = actor_yaw_values[actor_index]
        for timestep in range(overlap):
            if collision_steps[timestep]:
                continue
            collision_steps[timestep] = obb_overlap(
                center_a=ego[timestep],
                half_extents_a=ego_dim_values[timestep] / 2.0,
                yaw_a=float(ego_yaw_values[timestep]),
                center_b=actor[timestep],
                half_extents_b=dims[timestep] / 2.0,
                yaw_b=float(yaws[timestep]),
            )

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
    This metric is 2D only. For 3D TTC, supply a 3D AgentState and
    extend this function in a subclass.
    """
    ego = _coerce_agent_state(ego_state)
    actor = _coerce_agent_state(actor_state)

    relative_position = np.array([actor.x - ego.x, actor.y - ego.y], dtype=np.float64)
    relative_velocity = np.array([actor.vx - ego.vx, actor.vy - ego.vy], dtype=np.float64)
    radius = ego.radius + actor.radius

    return _solve_ttc_quadratic(relative_position, relative_velocity, radius)


def _solve_ttc_quadratic(
    relative_position: FloatArray,
    relative_velocity: FloatArray,
    radius: float,
) -> float:
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
    """Return minimum time-aligned Euclidean distance from ego to any actor."""
    ego = as_trajectory(ego_traj, name="ego_traj")
    actors = as_actor_trajectories(actor_trajs)
    if not actors:
        return inf

    min_distance = inf
    for actor in actors:
        overlap = min(ego.shape[0], actor.shape[0])
        if overlap == 0:
            continue
        distances = np.linalg.norm(
            _positions(ego[:overlap]) - _positions(actor[:overlap]),
            axis=1,
        )
        min_distance = min(min_distance, float(np.min(distances)))
    return min_distance


def _positions(arr: FloatArray) -> FloatArray:
    return arr


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


def _as_xy_trajectory(data: ArrayLike, *, name: str, allow_empty: bool) -> FloatArray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must be an Nx2 array")
    if arr.shape[0] == 0 and not allow_empty:
        raise ValueError(f"{name} must contain at least one point")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _as_yaw_array(data: object, *, name: str, length: int) -> FloatArray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != length:
        raise ValueError(f"{name} must be a length-{length} 1D array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _as_dims_array(data: object, *, name: str, length: int) -> FloatArray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.shape == (2,):
        dims = np.broadcast_to(arr, (length, 2)).astype(np.float64, copy=True)
    elif arr.ndim == 2 and arr.shape == (length, 2):
        dims = arr
    else:
        raise ValueError(f"{name} must have shape (2,) or ({length}, 2)")
    if not np.all(np.isfinite(dims)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(dims <= 0.0):
        raise ValueError(f"{name} must contain positive length and width values")
    return dims


def _as_actor_xy_list(actor_trajs: object) -> list[FloatArray]:
    if not isinstance(actor_trajs, list):
        raise ValueError("actor_trajs must be a list")
    return [
        _as_xy_trajectory(actor, name=f"actor_trajs[{index}]", allow_empty=True)
        for index, actor in enumerate(actor_trajs)
    ]


def _as_actor_dims_list(actor_dims: object, actors: list[FloatArray]) -> list[FloatArray]:
    if not isinstance(actor_dims, list):
        raise ValueError("actor_dims must be a list with one entry per actor")
    if len(actor_dims) != len(actors):
        raise ValueError("actor_dims must have the same length as actor_trajs")
    return [
        _as_dims_array(dims, name=f"actor_dims[{index}]", length=actor.shape[0])
        for index, (dims, actor) in enumerate(zip(actor_dims, actors))
    ]


def _as_actor_yaws_list(actor_yaws: object, actors: list[FloatArray]) -> list[FloatArray]:
    if not isinstance(actor_yaws, list):
        raise ValueError("actor_yaws must be a list with one entry per actor")
    if len(actor_yaws) != len(actors):
        raise ValueError("actor_yaws must have the same length as actor_trajs")
    return [
        _as_yaw_array(yaws, name=f"actor_yaws[{index}]", length=actor.shape[0])
        for index, (yaws, actor) in enumerate(zip(actor_yaws, actors))
    ]

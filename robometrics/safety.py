"""Geometry-based safety metrics."""

from __future__ import annotations

from collections.abc import Mapping
from math import inf, sqrt
from typing import Any, Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robometrics.geometry import (
    FloatArray,
    as_actor_trajectories,
    as_boolean_mask,
    as_numeric_array,
    as_trajectory,
    obb_overlap_batch,
    points_in_polygon,
    validate_nonnegative,
    validate_positive,
    validate_timestamps,
)
from robometrics.schemas import AgentState

_DEFAULT_FAILURE_CATEGORY_SEVERITY = {
    "minor": 1.0,
    "low": 1.0,
    "moderate": 2.0,
    "medium": 2.0,
    "major": 3.0,
    "high": 3.0,
    "critical": 4.0,
    "fatal": 5.0,
}


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
        collision_steps[:overlap] |= obb_overlap_batch(
            centers_a=ego[:overlap],
            half_extents_a=ego_dim_values[:overlap] / 2.0,
            yaws_a=ego_yaw_values[:overlap],
            centers_b=actor[:overlap],
            half_extents_b=dims[:overlap] / 2.0,
            yaws_b=yaws[:overlap],
        )

    if not np.any(covered_steps):
        return 0.0
    return float(np.mean(collision_steps[covered_steps]))


def time_to_collision(
    ego_state: Union[AgentState, ArrayLike, dict[str, Any]],
    actor_state: Union[AgentState, ArrayLike, dict[str, Any]],
    *,
    dt: Optional[float] = None,
) -> float:
    """Return constant-velocity time to collision for two disc agents.

    States must provide x, y, vx, and vy. Radius is optional and defaults to 0.
    When ``dt`` is supplied, Nx2/Nx3 trajectories are also accepted and the
    first segment is used to estimate each agent's constant velocity.
    A non-colliding or diverging pair returns math.inf.
    This metric is 2D only. For 3D TTC, supply a 3D AgentState and
    extend this function in a subclass.
    """
    if dt is None:
        ego = _coerce_agent_state(ego_state)
        actor = _coerce_agent_state(actor_state)
    else:
        ego = _coerce_ttc_state(ego_state, dt=dt, name="ego_state")
        actor = _coerce_ttc_state(actor_state, dt=dt, name="actor_state")

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


def recovery_success_rate(opportunities: ArrayLike, successes: ArrayLike) -> float:
    """Return successful recoveries divided by recovery opportunities."""
    opportunity_mask = as_boolean_mask(
        opportunities,
        name="opportunities",
        allow_empty=True,
    )
    success_mask = as_boolean_mask(successes, name="successes", allow_empty=True)
    if opportunity_mask.shape != success_mask.shape:
        raise ValueError("opportunities and successes must have the same shape")

    opportunity_count = int(np.count_nonzero(opportunity_mask))
    if opportunity_count == 0:
        return float("nan")
    success_count = int(np.count_nonzero(opportunity_mask & success_mask))
    return float(success_count / opportunity_count)


def failure_severity(
    failures: object,
    *,
    aggregation: Literal["mean", "max"] = "mean",
    category_scores: Optional[Mapping[str, float]] = None,
) -> float:
    """Return aggregated numeric severity for failure events."""
    if aggregation not in ("mean", "max"):
        raise ValueError("aggregation must be 'mean' or 'max'")

    severities = _coerce_severity_values(failures, category_scores)
    if severities.size == 0:
        return 0.0
    if aggregation == "max":
        return float(np.max(severities))
    return float(np.mean(severities))


def near_miss_rate(
    clearances: ArrayLike,
    threshold: float,
    collision_mask: Optional[ArrayLike] = None,
) -> float:
    """Return fraction of events that are near misses without collision."""
    clearance_arr = as_numeric_array(clearances, name="clearances")
    if clearance_arr.size == 0:
        raise ValueError("clearances must contain at least one value")
    limit = validate_positive(float(threshold), name="threshold")
    if collision_mask is None:
        collisions = np.zeros(clearance_arr.shape, dtype=np.bool_)
    else:
        collisions = as_boolean_mask(collision_mask, name="collision_mask")
        if collisions.shape != clearance_arr.shape:
            raise ValueError("collision_mask must have the same shape as clearances")

    near_misses = (clearance_arr < limit) & ~collisions
    return float(np.mean(near_misses))


def intervention_free_time(
    timestamps: ArrayLike,
    interventions: ArrayLike,
    *,
    mode: Literal["longest", "mean"] = "longest",
) -> float:
    """Return duration of intervention-free segments."""
    if mode not in ("longest", "mean"):
        raise ValueError("mode must be 'longest' or 'mean'")

    time_arr = validate_timestamps(timestamps)
    intervention_mask = as_boolean_mask(interventions, name="interventions")
    if intervention_mask.ndim != 1:
        raise ValueError("interventions must be a 1D array")
    if intervention_mask.shape != time_arr.shape:
        raise ValueError("timestamps and interventions must have the same shape")

    durations = _intervention_free_durations(time_arr, intervention_mask)
    if not durations:
        return 0.0
    if mode == "mean":
        return float(np.mean(np.asarray(durations, dtype=np.float64)))
    return float(max(durations))


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


def _coerce_ttc_state(
    state: Union[AgentState, ArrayLike, dict[str, Any]],
    *,
    dt: float,
    name: str,
) -> AgentState:
    timestep = validate_positive(float(dt), name="dt")
    if isinstance(state, (AgentState, dict)):
        return _coerce_agent_state(state)

    arr = np.asarray(state, dtype=np.float64)
    if arr.ndim == 1:
        return _coerce_agent_state(arr)
    traj = as_trajectory(arr, name=name)
    if traj.shape[0] < 2:
        raise ValueError(
            f"{name} trajectory input must contain at least two points when dt is set"
        )
    velocity = (traj[1, :2] - traj[0, :2]) / timestep
    return AgentState(
        x=float(traj[0, 0]),
        y=float(traj[0, 1]),
        vx=float(velocity[0]),
        vy=float(velocity[1]),
    )


def _coerce_severity_values(
    failures: object,
    category_scores: Optional[Mapping[str, float]],
) -> NDArray[np.float64]:
    raw = np.asarray(failures)
    if raw.size == 0:
        return np.empty(0, dtype=np.float64)

    try:
        numeric = np.asarray(failures, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return _severity_from_categories(raw, category_scores)

    if not np.all(np.isfinite(numeric)):
        raise ValueError("failures must contain only finite severity values")
    if np.any(numeric < 0.0):
        raise ValueError("failures must contain non-negative severity values")
    return np.asarray(numeric, dtype=np.float64)


def _severity_from_categories(
    raw: NDArray[Any],
    category_scores: Optional[Mapping[str, float]],
) -> NDArray[np.float64]:
    scores = dict(_DEFAULT_FAILURE_CATEGORY_SEVERITY)
    for key, value in (category_scores or {}).items():
        score = validate_nonnegative(float(value), name=f"category_scores[{key!r}]")
        scores[key.strip().lower()] = score

    values: list[float] = []
    for label in raw.reshape(-1):
        key = str(label).strip().lower()
        try:
            values.append(scores[key])
        except KeyError as exc:
            raise ValueError(f"unknown failure category: {label}") from exc
    return np.asarray(values, dtype=np.float64)


def _intervention_free_durations(
    timestamps: NDArray[np.float64],
    interventions: NDArray[np.bool_],
) -> list[float]:
    durations: list[float] = []
    start_index: Optional[int] = None
    for index, intervened in enumerate(interventions):
        if not intervened and start_index is None:
            start_index = index
        if start_index is not None and (intervened or index == interventions.shape[0] - 1):
            end_index = index - 1 if intervened else index
            durations.append(float(timestamps[end_index] - timestamps[start_index]))
            start_index = None
    return durations


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
        raise ValueError(
            f"actor_dims must have one entry per actor trajectory; "
            f"got {len(actor_dims)} dims entries for {len(actors)} actor_trajs"
        )
    return [
        _as_dims_array(dims, name=f"actor_dims[{index}]", length=actor.shape[0])
        for index, (dims, actor) in enumerate(zip(actor_dims, actors))
    ]


def _as_actor_yaws_list(actor_yaws: object, actors: list[FloatArray]) -> list[FloatArray]:
    if not isinstance(actor_yaws, list):
        raise ValueError("actor_yaws must be a list with one entry per actor")
    if len(actor_yaws) != len(actors):
        raise ValueError(
            f"actor_yaws must have one entry per actor trajectory; "
            f"got {len(actor_yaws)} yaw entries for {len(actors)} actor_trajs"
        )
    return [
        _as_yaw_array(yaws, name=f"actor_yaws[{index}]", length=actor.shape[0])
        for index, (yaws, actor) in enumerate(zip(actor_yaws, actors))
    ]

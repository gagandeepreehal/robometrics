"""Geometry helpers shared by RoboMetrics modules."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def as_trajectory(data: ArrayLike, *, name: str = "trajectory") -> FloatArray:
    """Return a finite non-empty Nx2 or Nx3 trajectory array."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError(f"{name} must be a Nx2 or Nx3 array")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one point")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def as_prediction_set(data: ArrayLike, *, name: str = "predictions") -> FloatArray:
    """Return a finite non-empty KxTx2 or KxTx3 prediction array."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] not in (2, 3):
        raise ValueError(f"{name} must be a KxTx2 or KxTx3 array")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one trajectory and one timestep")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def require_same_shape(
    left: FloatArray,
    right: FloatArray,
    left_name: str,
    right_name: str,
) -> None:
    """Validate that two arrays have identical shape."""
    if left.shape != right.shape:
        raise ValueError(f"{left_name} and {right_name} must have the same shape")


def require_same_time_and_dim(
    predictions: FloatArray,
    ground_truth: FloatArray,
    predictions_name: str = "predictions",
    ground_truth_name: str = "gt",
) -> None:
    """Validate that KxTxD predictions match TxD ground truth."""
    if predictions.shape[1:] != ground_truth.shape:
        raise ValueError(
            f"{predictions_name} timesteps/dimensions must match {ground_truth_name}"
        )


def xy(data: FloatArray) -> FloatArray:
    """Return XY columns from an Nx2/Nx3 trajectory-like array."""
    return data[:, :2]


def vector_norms(data: FloatArray) -> FloatArray:
    """Return row-wise Euclidean norms."""
    return np.asarray(np.linalg.norm(data, axis=1), dtype=np.float64)


def pointwise_distances(left: FloatArray, right: FloatArray) -> FloatArray:
    """Return pointwise Euclidean distances for equally shaped trajectories."""
    require_same_shape(left, right, "left", "right")
    return vector_norms(left - right)


def validate_positive(value: float, *, name: str) -> float:
    """Return value if positive, otherwise raise ValueError."""
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite value")
    return float(value)


def validate_nonnegative(value: float, *, name: str) -> float:
    """Return value if non-negative, otherwise raise ValueError."""
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a non-negative finite value")
    return float(value)


def as_actor_trajectories(actor_trajs: object) -> list[FloatArray]:
    """Normalize actor trajectories to a list of finite Nx2/Nx3 arrays."""
    try:
        arr = np.asarray(actor_trajs, dtype=np.float64)
    except (TypeError, ValueError):
        if not isinstance(actor_trajs, Iterable):
            raise ValueError("actor_trajs must be an actor trajectory or iterable") from None
        return [
            as_trajectory(actor, name=f"actor_trajs[{index}]")
            for index, actor in enumerate(actor_trajs)
        ]

    if arr.size == 0:
        return []
    if arr.ndim == 2:
        return [as_trajectory(arr, name="actor_trajs")]
    if arr.ndim == 3:
        return [
            as_trajectory(arr[index], name=f"actor_trajs[{index}]")
            for index in range(arr.shape[0])
        ]
    raise ValueError("actor_trajs must be Nx2/Nx3, MxNx2/MxNx3, or an iterable of trajectories")


def point_in_polygon(point: FloatArray, polygon: FloatArray, *, tolerance: float = 1e-9) -> bool:
    """Return True when a 2D point lies inside or on the boundary of a polygon."""
    px = float(point[0])
    py = float(point[1])
    poly = xy(polygon)
    if poly.shape[0] < 3:
        raise ValueError("lane_boundary must contain at least three polygon vertices")

    inside = False
    count = poly.shape[0]
    for index in range(count):
        start = poly[index]
        end = poly[(index + 1) % count]
        if _point_on_segment(np.array([px, py], dtype=np.float64), start, end, tolerance=tolerance):
            return True

        y_crosses = (start[1] > py) != (end[1] > py)
        if y_crosses:
            x_intersection = (end[0] - start[0]) * (py - start[1]) / (end[1] - start[1]) + start[0]
            if px < x_intersection:
                inside = not inside
    return inside


def points_in_polygon(points: FloatArray, polygon: FloatArray) -> NDArray[np.bool_]:
    """Return an inside-mask for points against a polygon."""
    return np.asarray([point_in_polygon(point, polygon) for point in xy(points)], dtype=np.bool_)


def _point_on_segment(
    point: FloatArray,
    start: FloatArray,
    end: FloatArray,
    *,
    tolerance: float,
) -> bool:
    segment = end - start
    point_delta = point - start
    squared_length = float(np.dot(segment, segment))
    if squared_length <= tolerance * tolerance:
        return bool(np.linalg.norm(point_delta) <= tolerance)

    cross = abs(float(segment[0] * point_delta[1] - segment[1] * point_delta[0]))
    if cross > tolerance:
        return False
    dot = float(np.dot(point_delta, segment))
    if dot < -tolerance:
        return False
    return dot <= squared_length + tolerance

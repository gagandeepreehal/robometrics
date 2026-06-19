"""Geometry helpers shared by RoboMetrics modules."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    FloatArray: TypeAlias = NDArray[np.float64]
else:
    FloatArray = Any


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


def as_1d_array(data: ArrayLike, *, name: str) -> FloatArray:
    """Return a finite non-empty 1D numeric array."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def as_boolean_mask(data: ArrayLike, *, name: str) -> NDArray[np.bool_]:
    """Return a non-empty 1D boolean mask from bool or 0/1 values."""
    raw = np.asarray(data)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if raw.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one value")
    if raw.dtype == np.bool_:
        return np.asarray(raw, dtype=np.bool_)

    arr = np.asarray(data, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.all((arr == 0.0) | (arr == 1.0)):
        raise ValueError(f"{name} must contain only boolean or 0/1 values")
    return np.asarray(arr.astype(np.bool_), dtype=np.bool_)


def as_points(data: ArrayLike, *, name: str) -> FloatArray:
    """Return a finite non-empty NxD point array."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a NxD array")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one point")
    if arr.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one coordinate")
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
    if _polygon_area(poly) <= tolerance:
        raise ValueError("lane_boundary must have non-zero polygon area")

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
    poly = xy(polygon)
    if _polygon_area(poly) <= 1e-9:
        raise ValueError("lane_boundary must have non-zero polygon area")
    return np.asarray([point_in_polygon(point, polygon) for point in xy(points)], dtype=np.bool_)


def obb_overlap(
    center_a: NDArray[np.float64],
    half_extents_a: NDArray[np.float64],
    yaw_a: float,
    center_b: NDArray[np.float64],
    half_extents_b: NDArray[np.float64],
    yaw_b: float,
) -> bool:
    """Return True if two 2D oriented bounding boxes overlap.

    Uses the Separating Axis Theorem (SAT) with four candidate axes:
    the two local X axes and two local Y axes of the two boxes.

    Inputs:
        center_*: (2,) center position in XY.
        half_extents_*: (2,) half-widths [half_length, half_width].
        yaw_*: heading angle in radians (rotation around Z).

    Reference: Gottschalk et al., OBBTree: A Hierarchical Structure for
               Rapid Interference Detection, SIGGRAPH 1996.
    """
    ca = _as_obb_vector(center_a, name="center_a")
    cb = _as_obb_vector(center_b, name="center_b")
    ea = _as_obb_vector(half_extents_a, name="half_extents_a")
    eb = _as_obb_vector(half_extents_b, name="half_extents_b")
    if np.any(ea < 0.0):
        raise ValueError("half_extents_a must contain non-negative values")
    if np.any(eb < 0.0):
        raise ValueError("half_extents_b must contain non-negative values")
    if not np.isfinite(yaw_a) or not np.isfinite(yaw_b):
        raise ValueError("yaw values must be finite")

    axes_a = _obb_axes(float(yaw_a))
    axes_b = _obb_axes(float(yaw_b))
    delta = cb - ca
    for axis in (axes_a[0], axes_a[1], axes_b[0], axes_b[1]):
        radius_a = _projected_radius(ea, axes_a, axis)
        radius_b = _projected_radius(eb, axes_b, axis)
        if abs(float(np.dot(delta, axis))) > radius_a + radius_b + 1e-12:
            return False
    return True


def _as_obb_vector(data: ArrayLike, *, name: str) -> NDArray[np.float64]:
    arr = np.asarray(data, dtype=np.float64)
    if arr.shape != (2,):
        raise ValueError(f"{name} must have shape (2,)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _obb_axes(yaw: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    cos_yaw = float(np.cos(yaw))
    sin_yaw = float(np.sin(yaw))
    return (
        np.array([cos_yaw, sin_yaw], dtype=np.float64),
        np.array([-sin_yaw, cos_yaw], dtype=np.float64),
    )


def _projected_radius(
    half_extents: NDArray[np.float64],
    axes: tuple[NDArray[np.float64], NDArray[np.float64]],
    projection_axis: NDArray[np.float64],
) -> float:
    return float(
        half_extents[0] * abs(float(np.dot(axes[0], projection_axis)))
        + half_extents[1] * abs(float(np.dot(axes[1], projection_axis)))
    )


def _polygon_area(poly: FloatArray) -> float:
    x_values = poly[:, 0]
    y_values = poly[:, 1]
    double_area = np.dot(x_values, np.roll(y_values, -1)) - np.dot(
        y_values,
        np.roll(x_values, -1),
    )
    return 0.5 * float(abs(double_area))


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

"""Optional ROS message adapters.

This module intentionally has no ROS imports at module import time.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from numpy.typing import NDArray

_ROS_HINT = "Install ROS 2 Humble or later and source the workspace to enable this adapter."


def from_path_msg(msg: Any) -> NDArray[np.float64]:
    """Convert a nav_msgs/Path ROS message to an Nx3 numpy array.

    Install ROS 2 Humble or later and source the workspace to enable this adapter.
    Extracts pose.position.x, pose.position.y, pose.position.z from each
    PoseStamped in msg.poses. Returns an Nx3 array.

    Raises:
        ImportError: if nav_msgs is not importable (ROS not installed)
        ValueError: if msg.poses is empty or missing
    """
    _ensure_ros_package("nav_msgs", msg)
    poses = _require_non_empty(getattr(msg, "poses", None), name="msg.poses")
    return _positions_to_array([pose_stamped.pose.position for pose_stamped in poses])


def from_odometry_sequence(msgs: list[Any]) -> NDArray[np.float64]:
    """Convert a list of nav_msgs/Odometry messages to an Nx3 array.

    Install ROS 2 Humble or later and source the workspace to enable this adapter.
    Extracts pose.pose.position.x/y/z from each message.

    Raises:
        ImportError: if nav_msgs is not importable
        ValueError: if msgs is empty
    """
    if not msgs:
        raise ValueError("msgs must contain at least one Odometry message")
    _ensure_ros_package("nav_msgs", msgs[0])
    return _positions_to_array([msg.pose.pose.position for msg in msgs])


def from_pose_array(msg: Any) -> NDArray[np.float64]:
    """Convert a geometry_msgs/PoseArray message to an Nx3 array.

    Install ROS 2 Humble or later and source the workspace to enable this adapter.
    Extracts position.x/y/z from each pose in msg.poses.

    Raises:
        ImportError: if geometry_msgs is not importable
        ValueError: if msg.poses is empty
    """
    _ensure_ros_package("geometry_msgs", msg)
    poses = _require_non_empty(getattr(msg, "poses", None), name="msg.poses")
    return _positions_to_array([pose.position for pose in poses])


def actor_trajs_from_marker_array(msg: Any) -> list[NDArray[np.float64]]:
    """Convert a visualization_msgs/MarkerArray to a list of trajectories.

    Install ROS 2 Humble or later and source the workspace to enable this adapter.
    Groups markers by marker.ns (namespace), sorts within each group by
    marker.id, and extracts marker.pose.position.x/y/z.
    Returns a list of Nx3 arrays, one per namespace.

    Raises:
        ImportError: if visualization_msgs is not importable
        ValueError: if msg.markers is empty
    """
    _ensure_ros_package("visualization_msgs", msg)
    markers = _require_non_empty(getattr(msg, "markers", None), name="msg.markers")
    by_namespace: dict[str, list[Any]] = defaultdict(list)
    for marker in markers:
        by_namespace[str(marker.ns)].append(marker)
    trajectories: list[NDArray[np.float64]] = []
    for namespace in sorted(by_namespace):
        ordered = sorted(by_namespace[namespace], key=lambda marker: int(marker.id))
        trajectories.append(_positions_to_array([marker.pose.position for marker in ordered]))
    return trajectories


def _ensure_ros_package(package: str, msg: Any) -> None:
    try:
        __import__(package)
    except ImportError as exc:
        if _is_plain_python_duck_type(msg, package):
            return
        raise ImportError(f"{package} is required for this ROS adapter. {_ROS_HINT}") from exc


def _is_plain_python_duck_type(msg: Any, package: str) -> bool:
    module_name = msg.__class__.__module__
    return not module_name.startswith(package)


def _require_non_empty(value: Any, *, name: str) -> list[Any]:
    if value is None:
        raise ValueError(f"{name} is required")
    items = list(value)
    if not items:
        raise ValueError(f"{name} must contain at least one item")
    return items


def _positions_to_array(positions: list[Any]) -> NDArray[np.float64]:
    points = [_position_to_point(position) for position in positions]
    return np.asarray(points, dtype=np.float64)


def _position_to_point(position: Any) -> list[float]:
    point = [float(position.x), float(position.y), float(position.z)]
    if not np.all(np.isfinite(point)):
        raise ValueError("ROS position values must be finite")
    return point

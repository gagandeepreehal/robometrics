from __future__ import annotations

import builtins
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from robometrics.ros import (
    actor_trajs_from_marker_array,
    from_odometry_sequence,
    from_path_msg,
    from_pose_array,
)


def _position(x: float, y: float, z: float) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


def _pose(x: float, y: float, z: float) -> SimpleNamespace:
    return SimpleNamespace(position=_position(x, y, z))


def test_from_path_msg_converts_pose_stamped_sequence() -> None:
    msg = SimpleNamespace(
        poses=[
            SimpleNamespace(pose=_pose(0.0, 0.0, 0.0)),
            SimpleNamespace(pose=_pose(1.0, 0.0, 0.5)),
            SimpleNamespace(pose=_pose(2.0, 0.0, 1.0)),
        ]
    )

    assert np.allclose(
        from_path_msg(msg),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.5], [2.0, 0.0, 1.0]]),
    )


def test_from_odometry_sequence_converts_messages() -> None:
    msgs = [
        SimpleNamespace(pose=SimpleNamespace(pose=_pose(0.0, 0.0, 0.0))),
        SimpleNamespace(pose=SimpleNamespace(pose=_pose(1.0, 2.0, 3.0))),
    ]

    assert np.allclose(from_odometry_sequence(msgs), np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]))


def test_from_pose_array_converts_poses() -> None:
    msg = SimpleNamespace(poses=[_pose(0.0, 0.0, 0.0), _pose(1.0, 2.0, 3.0)])

    assert np.allclose(from_pose_array(msg), np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]))


def test_actor_trajs_from_marker_array_groups_by_namespace() -> None:
    msg = SimpleNamespace(
        markers=[
            SimpleNamespace(ns="b", id=2, pose=_pose(2.0, 0.0, 0.0)),
            SimpleNamespace(ns="a", id=2, pose=_pose(1.0, 0.0, 0.0)),
            SimpleNamespace(ns="b", id=1, pose=_pose(0.0, 0.0, 0.0)),
            SimpleNamespace(ns="a", id=1, pose=_pose(0.0, 1.0, 0.0)),
        ]
    )

    trajectories = actor_trajs_from_marker_array(msg)

    assert len(trajectories) == 2
    assert np.allclose(trajectories[0], np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))
    assert np.allclose(trajectories[1], np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))


def test_ros_adapters_reject_empty_inputs() -> None:
    with pytest.raises(ValueError, match="msg.poses"):
        from_path_msg(SimpleNamespace(poses=[]))
    with pytest.raises(ValueError, match="msgs"):
        from_odometry_sequence([])
    with pytest.raises(ValueError, match="msg.poses"):
        from_pose_array(SimpleNamespace(poses=[]))
    with pytest.raises(ValueError, match="msg.markers"):
        actor_trajs_from_marker_array(SimpleNamespace(markers=[]))


def test_ros_import_error_guard_for_nav_msgs(monkeypatch) -> None:
    path_cls = type("Path", (), {"__module__": "nav_msgs.msg"})
    msg = path_cls()
    msg.poses = [SimpleNamespace(pose=_pose(0.0, 0.0, 0.0))]
    _patch_import_to_fail(monkeypatch, "nav_msgs")

    with pytest.raises(ImportError, match="Install ROS 2 Humble"):
        from_path_msg(msg)


def test_ros_import_error_guard_for_odometry(monkeypatch) -> None:
    odom_cls = type("Odometry", (), {"__module__": "nav_msgs.msg"})
    msg = odom_cls()
    msg.pose = SimpleNamespace(pose=_pose(0.0, 0.0, 0.0))
    _patch_import_to_fail(monkeypatch, "nav_msgs")

    with pytest.raises(ImportError, match="Install ROS 2 Humble"):
        from_odometry_sequence([msg])


def test_ros_import_error_guard_for_geometry_msgs(monkeypatch) -> None:
    pose_array_cls = type("PoseArray", (), {"__module__": "geometry_msgs.msg"})
    msg = pose_array_cls()
    msg.poses = [_pose(0.0, 0.0, 0.0)]
    _patch_import_to_fail(monkeypatch, "geometry_msgs")

    with pytest.raises(ImportError, match="Install ROS 2 Humble"):
        from_pose_array(msg)


def test_ros_import_error_guard_for_visualization_msgs(monkeypatch) -> None:
    marker_array_cls = type("MarkerArray", (), {"__module__": "visualization_msgs.msg"})
    msg = marker_array_cls()
    msg.markers = [SimpleNamespace(ns="a", id=1, pose=_pose(0.0, 0.0, 0.0))]
    _patch_import_to_fail(monkeypatch, "visualization_msgs")

    with pytest.raises(ImportError, match="Install ROS 2 Humble"):
        actor_trajs_from_marker_array(msg)


def _patch_import_to_fail(monkeypatch: pytest.MonkeyPatch, package: str) -> None:
    original_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == package or name.startswith(f"{package}."):
            raise ImportError(f"missing {package}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

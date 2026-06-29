"""ROS-style JSON adapters without importing ROS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

import numpy as np

from robometrics.schemas import Trajectory
from robometrics.validation import DatasetValidationResult, ValidationIssue, validate_dataset

PathLike = Union[str, Path]


class ROSStyleAdapter:
    """Load JSON exports that resemble ROS Path or PoseArray messages."""

    format_name = "ros-style-json"

    def load(self, path: PathLike) -> Trajectory:
        """Load a ROS-style JSON file into the standard Trajectory schema."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        points = _ros_points(payload)
        return Trajectory(points=points, metadata=self.metadata(path))

    def validate(self, path: PathLike) -> DatasetValidationResult:
        """Validate ROS-style JSON without requiring ROS packages."""
        resolved = Path(path)
        if resolved.suffix.lower() != ".json":
            return DatasetValidationResult(
                path=str(resolved),
                format=self.format_name,
                issues=[
                    ValidationIssue(
                        code="unsupported_format",
                        message="ROS-style adapter expects a JSON export",
                    )
                ],
            )
        try:
            self.load(resolved)
        except Exception as exc:  # noqa: BLE001 - validation reports loader failures.
            base = validate_dataset(resolved)
            issues = list(base.issues)
            issues.append(
                ValidationIssue(
                    code="adapter_error",
                    message=str(exc),
                )
            )
            return DatasetValidationResult(
                path=str(resolved),
                format=self.format_name,
                row_count=base.row_count,
                dimensions=base.dimensions,
                has_timestamps=base.has_timestamps,
                issues=issues,
                metadata=base.metadata,
            )
        return DatasetValidationResult(
            path=str(resolved),
            format=self.format_name,
            row_count=len(self.load(resolved).points),
            dimensions=len(self.load(resolved).points[0]),
            metadata=self.metadata(resolved),
        )

    def metadata(self, path: PathLike) -> dict[str, object]:
        """Return path-level adapter metadata."""
        resolved = Path(path)
        return {
            "adapter": self.__class__.__name__,
            "format": self.format_name,
            "path": str(resolved),
            "requires_ros": False,
        }


def _ros_points(payload: Any) -> list[list[float]]:
    if isinstance(payload, dict):
        if "poses" in payload:
            return [_position_from_pose_like(item) for item in payload["poses"]]
        if "trajectory" in payload or "points" in payload:
            records = payload.get("trajectory")
            if records is None:
                records = payload.get("points")
            if not isinstance(records, list):
                raise ValueError("trajectory or points must be a list")
            return [_generic_point(record) for record in records]
    if isinstance(payload, list):
        return [_generic_point(record) for record in payload]
    raise ValueError("ROS-style JSON must contain poses, trajectory, points, or a list")


def _position_from_pose_like(record: Any) -> list[float]:
    pose = record.get("pose", record) if isinstance(record, dict) else record
    position = pose.get("position", pose) if isinstance(pose, dict) else pose
    return _generic_point(position)


def _generic_point(record: Any) -> list[float]:
    if isinstance(record, dict):
        if "x" not in record or "y" not in record:
            raise ValueError("point records require x and y")
        point = [float(record["x"]), float(record["y"])]
        if "z" in record:
            point.append(float(record["z"]))
    elif isinstance(record, list):
        point = [float(value) for value in record]
    else:
        raise ValueError("point records must be objects or coordinate lists")
    arr = np.asarray(point, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] not in (2, 3) or not np.all(np.isfinite(arr)):
        raise ValueError("points must be finite 2D or 3D coordinates")
    return [float(value) for value in arr.tolist()]

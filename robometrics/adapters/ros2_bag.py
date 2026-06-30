"""ROS 2 bag JSON export adapter without importing rclpy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

from robometrics.schemas import Trajectory
from robometrics.validation import DatasetValidationResult, ValidationIssue

PathLike = Union[str, Path]


class ROS2BagJSONAdapter:
    """Load trajectory points from JSON exports of ROS 2 bag messages."""

    format_name = "ros2-bag-json"

    def __init__(self, *, topic: Optional[str] = None) -> None:
        self.topic = topic

    def load(self, path: PathLike) -> Trajectory:
        """Load a ROS 2 bag JSON export into the standard Trajectory schema."""
        resolved = Path(path)
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        points, timestamps, selected_topic = _points_from_ros2_export(
            payload, topic=self.topic
        )
        metadata = self.metadata(resolved)
        metadata["message_count"] = len(points)
        metadata["topic"] = selected_topic
        return Trajectory(
            points=points,
            timestamps=timestamps if len(timestamps) == len(points) else None,
            metadata=metadata,
        )

    def validate(self, path: PathLike) -> DatasetValidationResult:
        """Validate a ROS 2 bag JSON export without requiring ROS packages."""
        resolved = Path(path)
        if resolved.suffix.lower() != ".json":
            return DatasetValidationResult(
                path=str(resolved),
                format=self.format_name,
                issues=[
                    ValidationIssue(
                        code="unsupported_format",
                        message="ROS 2 bag adapter expects a JSON export",
                    )
                ],
                metadata=self.metadata(resolved),
            )
        try:
            trajectory = self.load(resolved)
        except Exception as exc:  # noqa: BLE001 - validation reports adapter failures.
            return DatasetValidationResult(
                path=str(resolved),
                format=self.format_name,
                issues=[ValidationIssue(code="adapter_error", message=str(exc))],
                metadata=self.metadata(resolved),
            )
        return DatasetValidationResult(
            path=str(resolved),
            format=self.format_name,
            row_count=len(trajectory.points),
            dimensions=len(trajectory.points[0]),
            has_timestamps=trajectory.timestamps is not None,
            issues=[],
            metadata=trajectory.metadata,
        )

    def metadata(self, path: PathLike) -> dict[str, object]:
        """Return path-level adapter metadata."""
        resolved = Path(path)
        return {
            "adapter": self.__class__.__name__,
            "format": self.format_name,
            "path": str(resolved),
            "exists": resolved.exists(),
            "topic": self.topic,
            "requires_ros": False,
            "requires_rclpy": False,
        }


def _points_from_ros2_export(
    payload: Any,
    *,
    topic: Optional[str],
) -> tuple[list[list[float]], list[float], Optional[str]]:
    from robometrics.adapters.ros_style import _generic_point, _ros_points

    records = _message_records(payload)
    if records is None:
        return _ros_points(_record_message(payload)), [], topic

    points: list[list[float]] = []
    timestamps: list[float] = []
    last_error: Optional[str] = None
    selected_topic = topic
    topic_bound = topic is not None
    for record in records:
        record_topic = _record_topic(record)
        if topic_bound and record_topic != selected_topic:
            continue
        message = _record_message(record)
        try:
            message_points = _ros_points(message)
        except ValueError:
            try:
                message_points = [_generic_point(_position_like(message))]
            except ValueError as exc:
                last_error = str(exc)
                continue
        if not topic_bound:
            selected_topic = record_topic
            topic_bound = True
        points.extend(message_points)
        timestamp = _record_timestamp(record)
        if timestamp is not None and len(message_points) == 1:
            timestamps.append(timestamp)

    if not points:
        topic_hint = f" for topic {topic!r}" if topic is not None else ""
        detail = f"; last message error: {last_error}" if last_error else ""
        raise ValueError(f"ROS 2 bag JSON contains no trajectory messages{topic_hint}{detail}")
    return points, timestamps, selected_topic


def _message_records(payload: Any) -> Optional[list[Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("messages", "records", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        if _looks_like_record(payload):
            return [payload]
    return None


def _looks_like_record(payload: dict[str, Any]) -> bool:
    return any(key in payload for key in ("topic", "topic_name", "message", "msg"))


def _record_topic(record: Any) -> Optional[str]:
    if isinstance(record, dict):
        for key in ("topic", "topic_name"):
            if key in record:
                return str(record[key])
    return None


def _record_message(record: Any) -> Any:
    if not isinstance(record, dict):
        return record
    for key in ("message", "msg", "data"):
        value = record.get(key)
        if isinstance(value, (dict, list)):
            return value
    return record


def _record_timestamp(record: Any) -> Optional[float]:
    if not isinstance(record, dict):
        return None
    for key in ("timestamp", "time", "t"):
        if key in record:
            return float(record[key])
    header = _record_message(record)
    if isinstance(header, dict):
        header_payload = header.get("header")
        stamp = header_payload.get("stamp") if isinstance(header_payload, dict) else None
        if isinstance(stamp, dict) and "sec" in stamp:
            return float(stamp["sec"]) + float(stamp.get("nanosec", 0.0)) / 1_000_000_000.0
    return None


def _position_like(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    if "pose" in payload and isinstance(payload["pose"], dict):
        return _position_like(payload["pose"])
    if "position" in payload and isinstance(payload["position"], dict):
        return payload["position"]
    if "translation" in payload and isinstance(payload["translation"], dict):
        return payload["translation"]
    return payload

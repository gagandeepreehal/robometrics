"""Optional MCAP adapter for JSON trajectory messages."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any, Optional, Union

from robometrics.schemas import Trajectory
from robometrics.validation import DatasetValidationResult, ValidationIssue

PathLike = Union[str, Path]

_MCAP_MESSAGE = (
    "MCAP loading requires the optional mcap extra. "
    "Install it with: pip install robometrics[mcap]"
)


class MCAPAdapter:
    """Load JSON trajectory messages from MCAP files.

    The core package does not depend on MCAP. Install ``robometrics[mcap]`` to
    enable this adapter. It decodes messages whose payload is UTF-8 JSON shaped
    like RoboMetrics trajectory JSON or ROS-style exported JSON. Binary ROS 2
    CDR decoding is intentionally out of scope for the lightweight adapter.
    """

    format_name = "mcap"

    def __init__(self, *, topic: Optional[str] = None) -> None:
        self.topic = topic

    def load(self, path: PathLike) -> Trajectory:
        """Load the first matching JSON trajectory stream from an MCAP file."""
        make_reader = _require_mcap_reader()
        resolved = Path(path)
        points: list[list[float]] = []
        metadata = self.metadata(resolved)
        last_error: Optional[str] = None
        selected_topic = self.topic
        topic_bound = self.topic is not None

        try:
            with resolved.open("rb") as handle:
                reader = make_reader(handle)
                for schema, channel, message in reader.iter_messages():
                    topic = str(getattr(channel, "topic", ""))
                    if topic_bound and topic != selected_topic:
                        continue
                    payload = _decode_json_message(getattr(message, "data", b""))
                    if payload is None:
                        continue
                    try:
                        message_points = _points_from_payload(payload)
                    except ValueError as exc:
                        last_error = str(exc)
                        continue
                    if not topic_bound:
                        selected_topic = topic
                        topic_bound = True
                    points.extend(message_points)
                    metadata.update(
                        {
                            "topic": selected_topic,
                            "message_encoding": str(getattr(channel, "message_encoding", "")),
                            "schema_name": str(getattr(schema, "name", "")),
                            "schema_encoding": str(getattr(schema, "encoding", "")),
                        }
                    )
        except OSError as exc:
            raise ValueError(f"could not read MCAP file {resolved}: {exc}") from exc

        if not points:
            detail = f"; last JSON error: {last_error}" if last_error else ""
            topic_hint = f" on topic {self.topic!r}" if self.topic is not None else ""
            raise ValueError(f"no JSON trajectory messages found in MCAP file{topic_hint}{detail}")
        metadata["point_count"] = len(points)
        return Trajectory(points=points, metadata=metadata)

    def validate(self, path: PathLike) -> DatasetValidationResult:
        """Validate an MCAP trajectory file when the optional reader is installed."""
        try:
            trajectory = self.load(path)
        except ImportError as exc:
            return DatasetValidationResult(
                path=str(path),
                format=self.format_name,
                issues=[
                    ValidationIssue(
                        code="optional_dependency_missing",
                        message=str(exc),
                    )
                ],
                metadata=self.metadata(path),
            )
        except Exception as exc:  # noqa: BLE001 - validation reports adapter failures.
            return DatasetValidationResult(
                path=str(path),
                format=self.format_name,
                issues=[ValidationIssue(code="adapter_error", message=str(exc))],
                metadata=self.metadata(path),
            )
        return DatasetValidationResult(
            path=str(path),
            format=self.format_name,
            row_count=len(trajectory.points),
            dimensions=len(trajectory.points[0]),
            issues=[],
            metadata=trajectory.metadata,
        )

    def metadata(self, path: PathLike) -> dict[str, object]:
        """Return path-level MCAP metadata."""
        resolved = Path(path)
        return {
            "adapter": self.__class__.__name__,
            "format": self.format_name,
            "path": str(resolved),
            "exists": resolved.exists(),
            "topic": self.topic,
            "requires_optional_dependency": "robometrics[mcap]",
        }


def _require_mcap_reader() -> Any:
    try:
        reader_module = import_module("mcap.reader")
    except ImportError as exc:
        raise ImportError(_MCAP_MESSAGE) from exc
    return reader_module.make_reader


def _decode_json_message(data: Any) -> Optional[Any]:
    if isinstance(data, memoryview):
        raw = data.tobytes()
    elif isinstance(data, bytearray):
        raw = bytes(data)
    elif isinstance(data, bytes):
        raw = data
    elif isinstance(data, str):
        raw = data.encode("utf-8")
    else:
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _points_from_payload(payload: Any) -> list[list[float]]:
    from robometrics.adapters.ros_style import _generic_point, _ros_points

    unwrapped = _unwrap_message(payload)
    try:
        return _ros_points(unwrapped)
    except ValueError:
        return [_generic_point(_position_like(unwrapped))]


def _unwrap_message(payload: Any) -> Any:
    current = payload
    while isinstance(current, dict):
        for key in ("message", "msg", "data"):
            nested = current.get(key)
            if isinstance(nested, (dict, list)):
                current = nested
                break
        else:
            return current
    return current


def _position_like(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    current = payload
    for key in ("pose", "position"):
        nested = current.get(key)
        if isinstance(nested, dict):
            current = nested
    if "pose" in current and isinstance(current["pose"], dict):
        return _position_like(current["pose"])
    if "position" in current and isinstance(current["position"], dict):
        return current["position"]
    return current

"""LeRobot-style lightweight adapter without importing LeRobot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

import numpy as np

from robometrics.schemas import Trajectory
from robometrics.validation import DatasetValidationResult, ValidationIssue, validate_dataset

PathLike = Union[str, Path]


class LeRobotStyleAdapter:
    """Load tiny JSON exports that contain state or observation trajectories."""

    format_name = "lerobot-style-json"

    def load(self, path: PathLike) -> Trajectory:
        """Load a LeRobot-style JSON file or directory into Trajectory."""
        resolved = _resolve_json_path(path)
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        points = _points_from_lerobot_payload(payload)
        return Trajectory(points=points, metadata=self.metadata(resolved))

    def validate(self, path: PathLike) -> DatasetValidationResult:
        """Validate the JSON file selected for this adapter."""
        try:
            resolved = _resolve_json_path(path)
            self.load(resolved)
        except Exception as exc:  # noqa: BLE001 - validation reports adapter failures.
            return DatasetValidationResult(
                path=str(path),
                format=self.format_name,
                issues=[ValidationIssue(code="adapter_error", message=str(exc))],
            )
        return validate_dataset(resolved)

    def metadata(self, path: PathLike) -> dict[str, object]:
        """Return path-level adapter metadata."""
        resolved = Path(path)
        return {
            "adapter": self.__class__.__name__,
            "format": self.format_name,
            "path": str(resolved),
            "requires_lerobot": False,
        }


def _resolve_json_path(path: PathLike) -> Path:
    resolved = Path(path)
    if resolved.is_dir():
        for name in ("trajectory.json", "episode.json", "data.json"):
            candidate = resolved / name
            if candidate.exists():
                return candidate
        raise ValueError(
            "LeRobot-style directory must contain trajectory.json, episode.json, or data.json"
        )
    return resolved


def _points_from_lerobot_payload(payload: Any) -> list[list[float]]:
    if isinstance(payload, dict):
        for key in ("trajectory", "points", "states"):
            if key in payload:
                return [_point(record) for record in payload[key]]
        if "steps" in payload:
            return [_point(_step_state(step)) for step in payload["steps"]]
    if isinstance(payload, list):
        return [_point(_step_state(item)) for item in payload]
    raise ValueError("LeRobot-style JSON must contain trajectory, points, states, or steps")


def _step_state(step: Any) -> Any:
    if not isinstance(step, dict):
        return step
    if "state" in step:
        return step["state"]
    observation = step.get("observation")
    if isinstance(observation, dict) and "state" in observation:
        return observation["state"]
    return step


def _point(record: Any) -> list[float]:
    if isinstance(record, dict):
        if "x" in record and "y" in record:
            raw_values: Any = [record["x"], record["y"]]
            if "z" in record:
                raw_values.append(record["z"])
        else:
            raw_values = record.get("position", record.get("ee_position"))
            if raw_values is None:
                raise ValueError("state records require x/y or position")
    else:
        raw_values = record
    arr = np.asarray(raw_values, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] < 2 or not np.all(np.isfinite(arr)):
        raise ValueError("state records must contain finite coordinates")
    return [float(value) for value in arr[:3].tolist()]

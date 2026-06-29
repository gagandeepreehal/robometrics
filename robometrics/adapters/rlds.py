"""RLDS-style lightweight JSON adapter without TensorFlow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

import numpy as np

from robometrics.schemas import Trajectory
from robometrics.validation import DatasetValidationResult, ValidationIssue

PathLike = Union[str, Path]


class RLDSStyleAdapter:
    """Load small JSON exports shaped like RLDS episodes."""

    format_name = "rlds-style-json"

    def load(self, path: PathLike) -> Trajectory:
        """Load an RLDS-style JSON export into Trajectory."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        points = _points_from_rlds(payload)
        return Trajectory(points=points, metadata=self.metadata(path))

    def validate(self, path: PathLike) -> DatasetValidationResult:
        """Validate an RLDS-style JSON export."""
        try:
            trajectory = self.load(path)
        except Exception as exc:  # noqa: BLE001 - validation reports adapter failures.
            return DatasetValidationResult(
                path=str(path),
                format=self.format_name,
                issues=[ValidationIssue(code="adapter_error", message=str(exc))],
            )
        return DatasetValidationResult(
            path=str(path),
            format=self.format_name,
            row_count=len(trajectory.points),
            dimensions=len(trajectory.points[0]),
            issues=[],
            metadata=self.metadata(path),
        )

    def metadata(self, path: PathLike) -> dict[str, object]:
        """Return path-level adapter metadata."""
        resolved = Path(path)
        return {
            "adapter": self.__class__.__name__,
            "format": self.format_name,
            "path": str(resolved),
            "requires_tensorflow": False,
        }


def _points_from_rlds(payload: Any) -> list[list[float]]:
    if isinstance(payload, dict):
        if "episodes" in payload:
            episodes = payload["episodes"]
            if not episodes:
                raise ValueError("RLDS-style JSON contains no episodes")
            return _points_from_rlds(episodes[0])
        if "steps" in payload:
            return [_point(_observation(step)) for step in payload["steps"]]
        if "trajectory" in payload:
            return [_point(record) for record in payload["trajectory"]]
    if isinstance(payload, list):
        return [_point(_observation(step)) for step in payload]
    raise ValueError("RLDS-style JSON must contain episodes, steps, trajectory, or a step list")


def _observation(step: Any) -> Any:
    if not isinstance(step, dict):
        return step
    observation = step.get("observation", step)
    if isinstance(observation, dict):
        for key in ("state", "position", "ee_position"):
            if key in observation:
                return observation[key]
    return observation


def _point(record: Any) -> list[float]:
    if isinstance(record, dict):
        if "x" in record and "y" in record:
            values = [record["x"], record["y"]]
            if "z" in record:
                values.append(record["z"])
        else:
            raise ValueError("observation records require x/y or a numeric state")
    else:
        values = record
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] < 2 or not np.all(np.isfinite(arr)):
        raise ValueError("observation records must contain finite coordinates")
    return [float(value) for value in arr[:3].tolist()]

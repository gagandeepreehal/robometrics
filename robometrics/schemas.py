"""Small typed schemas for RoboMetrics inputs and outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from robometrics.results import EvaluationResult, MetricResult

__all__ = ["AgentState", "EvaluationResult", "MetricResult", "Trajectory"]


@dataclass
class Trajectory:
    """Serializable trajectory with optional timestamps.

    ``points`` must be a finite, non-empty ``Nx2`` or ``Nx3`` sequence in meters.
    ``timestamps`` must be finite seconds and match the point count when provided.
    """

    points: list[list[float]]
    timestamps: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            arr = np.asarray(self.points, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("points must be a non-empty Nx2 or Nx3 list") from exc
        if arr.ndim != 2 or arr.shape[1] not in (2, 3):
            raise ValueError("points must be a non-empty Nx2 or Nx3 list")
        if arr.shape[0] == 0:
            raise ValueError("points must contain at least one point")
        if not np.all(np.isfinite(arr)):
            raise ValueError("points must contain only finite values")

        if self.timestamps is not None and len(self.timestamps) != len(self.points):
            raise ValueError("timestamps length must match points length")
        if self.timestamps is not None:
            timestamps = np.asarray(self.timestamps, dtype=np.float64)
            if timestamps.ndim != 1 or not np.all(np.isfinite(timestamps)):
                raise ValueError("timestamps must be a one-dimensional finite list")

        self.points = arr.tolist()
        self.timestamps = None if self.timestamps is None else timestamps.tolist()
        self.metadata = dict(self.metadata)

    def array(self) -> NDArray[np.float64]:
        """Return trajectory points as a NumPy array."""
        return np.asarray(self.points, dtype=np.float64)


@dataclass
class AgentState:
    """Planar constant-velocity agent state.

    Position is in meters, velocity is in meters per second, heading is in radians,
    and radius is in meters.
    """

    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    heading: float | None = None
    radius: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("x", "y", "vx", "vy", "heading", "radius"):
            value = getattr(self, name)
            if value is not None and not np.isfinite(value):
                raise ValueError("agent state values must be finite")
        if self.radius < 0:
            raise ValueError("radius must be non-negative")
        self.x = float(self.x)
        self.y = float(self.y)
        self.vx = float(self.vx)
        self.vy = float(self.vy)
        self.heading = None if self.heading is None else float(self.heading)
        self.radius = float(self.radius)
        self.metadata = dict(self.metadata)

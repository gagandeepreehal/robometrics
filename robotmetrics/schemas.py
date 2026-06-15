"""Pydantic schemas for RobotMetrics inputs and outputs."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator, model_validator

from robotmetrics.results import EvaluationResult, MetricResult

__all__ = ["AgentState", "EvaluationResult", "MetricResult", "Trajectory"]


class Trajectory(BaseModel):
    """Serializable trajectory schema with optional timestamps."""

    points: list[list[float]]
    timestamps: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("points")
    @classmethod
    def _validate_points(cls, points: list[list[float]]) -> list[list[float]]:
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] not in (2, 3):
            raise ValueError("points must be a non-empty Nx2 or Nx3 list")
        if arr.shape[0] == 0:
            raise ValueError("points must contain at least one point")
        if not np.all(np.isfinite(arr)):
            raise ValueError("points must contain only finite values")
        return points

    @field_validator("timestamps")
    @classmethod
    def _validate_timestamps(cls, timestamps: list[float] | None) -> list[float] | None:
        if timestamps is None:
            return timestamps
        arr = np.asarray(timestamps, dtype=np.float64)
        if arr.ndim != 1 or not np.all(np.isfinite(arr)):
            raise ValueError("timestamps must be a one-dimensional finite list")
        return timestamps

    @model_validator(mode="after")
    def _validate_lengths(self) -> Trajectory:
        if self.timestamps is not None and len(self.timestamps) != len(self.points):
            raise ValueError("timestamps length must match points length")
        return self

    def array(self) -> NDArray[np.float64]:
        """Return trajectory points as a NumPy array."""
        return np.asarray(self.points, dtype=np.float64)


class AgentState(BaseModel):
    """Planar constant-velocity agent state."""

    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    heading: float | None = None
    radius: float = Field(default=0.0, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("x", "y", "vx", "vy", "heading", "radius")
    @classmethod
    def _finite_values(cls, value: float | None) -> float | None:
        if value is not None and not np.isfinite(value):
            raise ValueError("agent state values must be finite")
        return value

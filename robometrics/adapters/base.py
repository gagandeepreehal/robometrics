"""Adapter interfaces for lightweight external dataset formats."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Union

from robometrics.schemas import Trajectory
from robometrics.validation import DatasetValidationResult

PathLike = Union[str, Path]


class TrajectoryAdapter(Protocol):
    """Consistent interface for dataset adapters."""

    def load(self, path: PathLike) -> Trajectory:
        """Load a path into the standard Trajectory schema."""

    def validate(self, path: PathLike) -> DatasetValidationResult:
        """Validate a path without requiring heavy optional dependencies."""

    def metadata(self, path: PathLike) -> dict[str, object]:
        """Return lightweight metadata for a path."""

"""MCAP adapter placeholder with an explicit optional dependency error."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from robometrics.schemas import Trajectory
from robometrics.validation import DatasetValidationResult, ValidationIssue

PathLike = Union[str, Path]

_MCAP_MESSAGE = (
    "MCAP loading is not part of the lightweight core install. "
    "Install the mcap Python package and use a project-specific adapter, or "
    "export trajectories to CSV/JSON before calling RoboMetrics."
)


class MCAPAdapter:
    """Placeholder adapter for MCAP trajectory logs."""

    format_name = "mcap"

    def load(self, path: PathLike) -> Trajectory:
        """Raise an explicit optional-dependency error."""
        raise ImportError(_MCAP_MESSAGE)

    def validate(self, path: PathLike) -> DatasetValidationResult:
        """Return a validation report that explains MCAP is optional."""
        return DatasetValidationResult(
            path=str(path),
            format=self.format_name,
            issues=[ValidationIssue(code="optional_dependency_missing", message=_MCAP_MESSAGE)],
            metadata=self.metadata(path),
        )

    def metadata(self, path: PathLike) -> dict[str, object]:
        """Return path-level MCAP metadata."""
        return {
            "adapter": self.__class__.__name__,
            "format": self.format_name,
            "path": str(Path(path)),
            "requires_optional_dependency": "mcap",
        }

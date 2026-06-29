"""Generic CSV and JSON trajectory adapters."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Union

from robometrics.io import load_trajectory_csv, load_trajectory_json
from robometrics.schemas import Trajectory
from robometrics.validation import DatasetValidationResult, validate_dataset

PathLike = Union[str, Path]


class GenericCSVAdapter:
    """Load CSV trajectory files with x/y and optional z columns."""

    format_name = "csv"

    def load(self, path: PathLike) -> Trajectory:
        """Load a CSV trajectory into the standard Trajectory schema."""
        arr = load_trajectory_csv(path, z_col="z" if _csv_has_z_column(path) else None)
        return Trajectory(points=arr.tolist(), metadata=self.metadata(path))

    def validate(self, path: PathLike) -> DatasetValidationResult:
        """Validate a CSV trajectory file."""
        return validate_dataset(path)

    def metadata(self, path: PathLike) -> dict[str, object]:
        """Return path-level CSV metadata."""
        resolved = Path(path)
        return {
            "adapter": self.__class__.__name__,
            "format": self.format_name,
            "path": str(resolved),
            "exists": resolved.exists(),
        }


def _csv_has_z_column(path: PathLike) -> bool:
    try:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
    except OSError:
        return False
    return "z" in header


class GenericJSONAdapter:
    """Load JSON trajectory files with trajectory or points records."""

    format_name = "json"

    def load(self, path: PathLike) -> Trajectory:
        """Load a JSON trajectory into the standard Trajectory schema."""
        arr = load_trajectory_json(path)
        return Trajectory(points=arr.tolist(), metadata=self.metadata(path))

    def validate(self, path: PathLike) -> DatasetValidationResult:
        """Validate a JSON trajectory file."""
        return validate_dataset(path)

    def metadata(self, path: PathLike) -> dict[str, object]:
        """Return path-level JSON metadata."""
        resolved = Path(path)
        return {
            "adapter": self.__class__.__name__,
            "format": self.format_name,
            "path": str(resolved),
            "exists": resolved.exists(),
        }

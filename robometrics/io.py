"""Simple trajectory loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from robometrics.geometry import FloatArray, as_trajectory


class TrajectoryIOError(ValueError):
    """Raised when a trajectory file cannot be read or parsed."""


def load_numpy(source: str | Path | ArrayLike) -> FloatArray:
    """Load a trajectory from an in-memory array or .npy/.npz file."""
    if isinstance(source, str | Path):
        path = Path(source)
        if not path.exists():
            raise TrajectoryIOError(f"trajectory file does not exist: {path}")
        try:
            if path.suffix == ".npz":
                with np.load(path) as data:
                    if "trajectory" in data:
                        arr = data["trajectory"]
                    else:
                        keys = list(data.keys())
                        if not keys:
                            raise ValueError("npz file does not contain arrays")
                        arr = data[keys[0]]
            else:
                arr = np.load(path)
        except OSError as exc:
            raise TrajectoryIOError(f"could not read NumPy trajectory file {path}: {exc}") from exc
    else:
        arr = np.asarray(source, dtype=np.float64)
    return as_trajectory(arr, name="trajectory")


def load_csv(
    path: str | Path,
    *,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str | None = None,
) -> FloatArray:
    """Load a trajectory from a CSV file with x/y columns and optional z."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise TrajectoryIOError(f"trajectory file does not exist: {csv_path}")
    try:
        frame = pd.read_csv(csv_path)
    except (OSError, pd.errors.ParserError) as exc:
        raise TrajectoryIOError(f"could not read CSV trajectory file {csv_path}: {exc}") from exc
    columns = [x_col, y_col] if z_col is None else [x_col, y_col, z_col]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"CSV file is missing required columns: {missing}")
    return as_trajectory(frame[columns].to_numpy(dtype=np.float64), name="trajectory")


def load_trajectory_csv(
    path: str | Path,
    *,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str | None = None,
) -> FloatArray:
    """Load a finite ``Nx2`` or ``Nx3`` trajectory from a CSV file."""
    return load_csv(path, x_col=x_col, y_col=y_col, z_col=z_col)


def load_json(path: str | Path) -> FloatArray:
    """Load a trajectory from JSON.

    Supported format:
    {"trajectory": [{"t": 0.0, "x": 0.0, "y": 0.0}, ...]}
    A top-level list of point objects is also accepted.
    """
    json_path = Path(path)
    if not json_path.exists():
        raise TrajectoryIOError(f"trajectory file does not exist: {json_path}")
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TrajectoryIOError(f"could not parse JSON trajectory file {json_path}: {exc}") from exc
    except OSError as exc:
        raise TrajectoryIOError(f"could not read JSON trajectory file {json_path}: {exc}") from exc
    records = (
        payload["trajectory"]
        if isinstance(payload, dict) and "trajectory" in payload
        else payload
    )
    if not isinstance(records, list):
        raise ValueError("JSON trajectory must be a list or contain a 'trajectory' list")

    points: list[list[float]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"trajectory record {index} must be an object")
        if "x" not in record or "y" not in record:
            raise ValueError(f"trajectory record {index} must contain x and y")
        point = [float(record["x"]), float(record["y"])]
        if "z" in record:
            point.append(float(record["z"]))
        points.append(point)
    return as_trajectory(points, name="trajectory")


def load_trajectory_json(path: str | Path) -> FloatArray:
    """Load a finite ``Nx2`` or ``Nx3`` trajectory from a JSON file."""
    return load_json(path)


def load_trajectory(path: str | Path) -> FloatArray:
    """Load a trajectory from .npy, .npz, .csv, or .json."""
    trajectory_path = Path(path)
    suffix = trajectory_path.suffix.lower()
    if suffix in {".npy", ".npz"}:
        return load_numpy(trajectory_path)
    if suffix == ".csv":
        return load_csv(trajectory_path)
    if suffix == ".json":
        return load_json(trajectory_path)
    raise ValueError(f"unsupported trajectory file extension: {suffix}")


def trajectory_to_json_records(traj: ArrayLike) -> list[dict[str, Any]]:
    """Convert a trajectory array to JSON-compatible point records."""
    arr = as_trajectory(traj, name="traj")
    records: list[dict[str, Any]] = []
    for point in arr:
        record: dict[str, Any] = {"x": float(point[0]), "y": float(point[1])}
        if point.shape[0] == 3:
            record["z"] = float(point[2])
        records.append(record)
    return records

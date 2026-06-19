"""Generic trajectory batch loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import NDArray

from robometrics.io import (
    TrajectoryIOError,
    load_trajectory_csv,
    load_trajectory_json,
)


def load_trajectory_batch(
    paths: list[Union[str, Path]],
    *,
    format: str = "auto",
) -> list[NDArray[np.float64]]:
    """Load a list of trajectory files and return as a list of arrays.

    Supported formats: "csv", "json", "auto" (infer from extension).
    Delegates to load_trajectory_csv and load_trajectory_json in io.py.
    Returns trajectories in the same order as paths.
    Raises TrajectoryIOError for any file that fails to load.
    """
    return [_load_one(path, format=format) for path in paths]


def trajectories_to_dataset(
    ego_paths: list[Union[str, Path]],
    gt_paths: list[Union[str, Path]],
    *,
    format: str = "auto",
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Load matched ego prediction and ground-truth trajectory files.

    Returns (predictions, ground_truths) lists of equal length, suitable
    for passing directly to Evaluator.evaluate_dataset().
    Raises ValueError if ego_paths and gt_paths have different lengths.
    Raises TrajectoryIOError for any file that fails to load.
    """
    if len(ego_paths) != len(gt_paths):
        raise ValueError("ego_paths and gt_paths must have the same length")
    return (
        load_trajectory_batch(ego_paths, format=format),
        load_trajectory_batch(gt_paths, format=format),
    )


def _load_one(path: Union[str, Path], *, format: str) -> NDArray[np.float64]:
    resolved_format = _resolve_format(path, format=format)
    try:
        if resolved_format == "csv":
            return load_trajectory_csv(path)
        if resolved_format == "json":
            return load_trajectory_json(path)
    except TrajectoryIOError:
        raise
    except ValueError as exc:
        raise TrajectoryIOError(f"could not load trajectory file {path}: {exc}") from exc
    raise ValueError("format must be 'csv', 'json', or 'auto'")


def _resolve_format(path: Union[str, Path], *, format: str) -> str:
    normalized = format.lower()
    if normalized in {"csv", "json"}:
        return normalized
    if normalized != "auto":
        raise ValueError("format must be 'csv', 'json', or 'auto'")
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".json":
        return "json"
    raise TrajectoryIOError(f"unsupported trajectory file extension: {suffix}")

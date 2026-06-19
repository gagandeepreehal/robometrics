from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from robometrics.io import (
    TrajectoryIOError,
    load_csv,
    load_json,
    load_numpy,
    load_trajectory,
    load_trajectory_csv,
    load_trajectory_json,
    trajectory_to_json_records,
)
from robometrics.schemas import EvaluationResult, MetricResult, Trajectory


def test_load_numpy_from_array_and_file(tmp_path) -> None:
    traj = np.array([[0.0, 0.0], [1.0, 0.0]])
    path = tmp_path / "traj.npy"
    np.save(path, traj)

    assert np.allclose(load_numpy(traj), traj)
    assert np.allclose(load_numpy(path), traj)
    assert np.allclose(load_trajectory(path), traj)


def test_load_csv_and_json(tmp_path) -> None:
    traj = np.array([[0.0, 0.0], [0.2, 0.0]])
    csv_path = tmp_path / "traj.csv"
    json_path = tmp_path / "traj.json"

    pd.DataFrame({"t": [0.0, 0.1], "x": [0.0, 0.2], "y": [0.0, 0.0]}).to_csv(
        csv_path,
        index=False,
    )
    json_path.write_text(
        json.dumps(
            {
                "trajectory": [
                    {"t": 0.0, "x": 0.0, "y": 0.0},
                    {"t": 0.1, "x": 0.2, "y": 0.0},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert np.allclose(load_csv(csv_path), traj)
    assert np.allclose(load_json(json_path), traj)
    assert np.allclose(load_trajectory(json_path), traj)
    assert np.allclose(load_trajectory_csv(csv_path), traj)
    assert np.allclose(load_trajectory_json(json_path), traj)


def test_loaders_raise_trajectory_io_error_for_file_failures(tmp_path) -> None:
    missing_path = tmp_path / "missing.json"
    malformed_json_path = tmp_path / "bad.json"
    malformed_json_path.write_text("{bad json", encoding="utf-8")

    with pytest.raises(TrajectoryIOError, match="trajectory file does not exist"):
        load_trajectory(missing_path)

    with pytest.raises(TrajectoryIOError, match="could not parse JSON trajectory file"):
        load_json(malformed_json_path)


def test_load_csv_missing_columns_stays_validation_error(tmp_path) -> None:
    csv_path = tmp_path / "missing_xy.csv"
    pd.DataFrame({"x": [0.0]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="CSV file is missing required columns"):
        load_csv(csv_path)


def test_trajectory_to_json_records() -> None:
    records = trajectory_to_json_records(np.array([[0.0, 1.0, 2.0]]))

    assert records == [{"x": 0.0, "y": 1.0, "z": 2.0}]


def test_schemas_validate_and_aggregate() -> None:
    trajectory = Trajectory(points=[[0.0, 0.0], [1.0, 0.0]], timestamps=[0.0, 0.1])
    metric = MetricResult(name="example", value=1.0, unit="m", passed=True, threshold=2.0)
    result = EvaluationResult(metrics=[metric])

    assert np.allclose(trajectory.array(), np.array([[0.0, 0.0], [1.0, 0.0]]))
    assert result.passed is True


def test_schema_rejects_bad_trajectory() -> None:
    with pytest.raises(ValueError):
        Trajectory(points=[[0.0, float("nan")]])

from __future__ import annotations

import json

import numpy as np
import pytest

from robometrics.io import TrajectoryIOError
from robometrics.loaders import (
    load_nuscenes_trajectories,
    load_trajectory_batch,
    trajectories_to_dataset,
)


def _write_nuscenes_fixture(path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "token": "ann-1",
                    "sample_token": "sample-1",
                    "instance_token": "actor-a",
                    "translation": [0.0, 0.0, 0.0],
                },
                {
                    "token": "ann-2",
                    "sample_token": "sample-2",
                    "instance_token": "actor-b",
                    "translation": [10.0, 0.0, 0.0],
                },
                {
                    "token": "ann-3",
                    "sample_token": "sample-3",
                    "instance_token": "actor-a",
                    "translation": [1.0, 0.0, 0.0],
                },
            ]
        ),
        encoding="utf-8",
    )


def test_load_nuscenes_trajectories_groups_by_instance_token(tmp_path) -> None:
    path = tmp_path / "sample_annotation.json"
    _write_nuscenes_fixture(path)

    trajectories = load_nuscenes_trajectories(path)

    assert sorted(trajectories) == ["actor-a", "actor-b"]
    assert trajectories["actor-a"].shape == (2, 3)
    assert np.allclose(trajectories["actor-a"], np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    assert np.allclose(trajectories["actor-b"], np.array([[10.0, 0.0, 0.0]]))


def test_load_nuscenes_trajectories_filters_instance_token(tmp_path) -> None:
    path = tmp_path / "sample_annotation.json"
    _write_nuscenes_fixture(path)

    trajectories = load_nuscenes_trajectories(path, instance_token="actor-b")

    assert list(trajectories) == ["actor-b"]
    assert np.allclose(trajectories["actor-b"], np.array([[10.0, 0.0, 0.0]]))


def test_load_nuscenes_trajectories_sorts_by_timestamp(tmp_path) -> None:
    path = tmp_path / "sample_annotation.json"
    path.write_text(
        json.dumps(
            [
                {
                    "instance_token": "actor-a",
                    "timestamp": 300,
                    "translation": [3.0, 0.0, 0.0],
                },
                {
                    "instance_token": "actor-a",
                    "timestamp": 100,
                    "translation": [1.0, 0.0, 0.0],
                },
                {
                    "instance_token": "actor-a",
                    "timestamp": 200,
                    "translation": [2.0, 0.0, 0.0],
                },
            ]
        ),
        encoding="utf-8",
    )

    trajectories = load_nuscenes_trajectories(path)

    assert np.allclose(
        trajectories["actor-a"],
        np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
    )


def test_load_nuscenes_trajectories_accepts_string_timestamps_and_tie_breaks(tmp_path) -> None:
    path = tmp_path / "sample_annotation.json"
    path.write_text(
        json.dumps(
            [
                {
                    "instance_token": "actor-a",
                    "timestamp": "1.0",
                    "translation": [2.0, 0.0, 0.0],
                },
                {
                    "instance_token": "actor-a",
                    "timestamp": "1.0",
                    "translation": [3.0, 0.0, 0.0],
                },
            ]
        ),
        encoding="utf-8",
    )

    trajectories = load_nuscenes_trajectories(path)

    assert np.allclose(
        trajectories["actor-a"],
        np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
    )


def test_load_nuscenes_trajectories_missing_file_raises(tmp_path) -> None:
    with pytest.raises(TrajectoryIOError, match="does not exist"):
        load_nuscenes_trajectories(tmp_path / "missing.json")


def test_load_nuscenes_trajectories_rejects_invalid_json(tmp_path) -> None:
    path = tmp_path / "sample_annotation.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(TrajectoryIOError, match="could not parse"):
        load_nuscenes_trajectories(path)


def test_load_nuscenes_trajectories_rejects_non_list_payload(tmp_path) -> None:
    path = tmp_path / "sample_annotation.json"
    path.write_text(json.dumps({"records": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="must be a list"):
        load_nuscenes_trajectories(path)


def test_load_nuscenes_trajectories_rejects_non_object_records(tmp_path) -> None:
    path = tmp_path / "sample_annotation.json"
    path.write_text(json.dumps([1]), encoding="utf-8")

    with pytest.raises(ValueError, match="record 0 must be an object"):
        load_nuscenes_trajectories(path)


def test_load_nuscenes_trajectories_missing_instance_token_raises(tmp_path) -> None:
    path = tmp_path / "sample_annotation.json"
    path.write_text(json.dumps([{"translation": [0.0, 0.0, 0.0]}]), encoding="utf-8")

    with pytest.raises(ValueError, match="instance_token"):
        load_nuscenes_trajectories(path)


def test_load_nuscenes_trajectories_missing_translation_raises(tmp_path) -> None:
    path = tmp_path / "sample_annotation.json"
    path.write_text(
        json.dumps([{"token": "ann-1", "sample_token": "sample-1", "instance_token": "actor-a"}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="translation"):
        load_nuscenes_trajectories(path)


def test_load_nuscenes_trajectories_rejects_bad_translation_values(tmp_path) -> None:
    path = tmp_path / "sample_annotation.json"
    path.write_text(
        json.dumps(
            [
                {"instance_token": "actor-a", "translation": ["bad", 0.0, 0.0]},
                {"instance_token": "actor-b", "translation": [float("inf"), 0.0, 0.0]},
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="translation must be numeric"):
        load_nuscenes_trajectories(path, instance_token="actor-a")
    with pytest.raises(ValueError, match="translation must be finite"):
        load_nuscenes_trajectories(path, instance_token="actor-b")


@pytest.mark.parametrize("timestamp", [[], "not-a-number", float("inf")])
def test_load_nuscenes_trajectories_rejects_bad_timestamps(tmp_path, timestamp) -> None:
    path = tmp_path / "sample_annotation.json"
    path.write_text(
        json.dumps(
            [
                {
                    "instance_token": "actor-a",
                    "timestamp": timestamp,
                    "translation": [0.0, 0.0, 0.0],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="timestamp must be"):
        load_nuscenes_trajectories(path)


def test_load_trajectory_batch_loads_csv_files(tmp_path) -> None:
    pd = pytest.importorskip("pandas")
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 0.0]}).to_csv(first, index=False)
    pd.DataFrame({"x": [2.0, 3.0], "y": [0.0, 0.0]}).to_csv(second, index=False)

    trajectories = load_trajectory_batch([first, second], format="csv")

    assert len(trajectories) == 2
    assert np.allclose(trajectories[0], np.array([[0.0, 0.0], [1.0, 0.0]]))
    assert np.allclose(trajectories[1], np.array([[2.0, 0.0], [3.0, 0.0]]))


def test_trajectories_to_dataset_loads_matched_json_files(tmp_path) -> None:
    pred = tmp_path / "pred.json"
    gt = tmp_path / "gt.json"
    pred.write_text(json.dumps({"points": [[0.0, 0.0], [1.0, 0.0]]}), encoding="utf-8")
    gt.write_text(json.dumps({"points": [[0.0, 0.0], [1.0, 0.5]]}), encoding="utf-8")

    predictions, ground_truths = trajectories_to_dataset([pred], [gt])

    assert np.allclose(predictions[0], np.array([[0.0, 0.0], [1.0, 0.0]]))
    assert np.allclose(ground_truths[0], np.array([[0.0, 0.0], [1.0, 0.5]]))


def test_trajectories_to_dataset_rejects_mismatched_lengths(tmp_path) -> None:
    with pytest.raises(ValueError, match="same length"):
        trajectories_to_dataset([tmp_path / "a.csv"], [])


def test_load_trajectory_batch_auto_format_inference(tmp_path) -> None:
    pd = pytest.importorskip("pandas")
    csv_path = tmp_path / "traj.csv"
    json_path = tmp_path / "traj.json"
    pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 0.0]}).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({"points": [[2.0, 0.0], [3.0, 0.0]]}), encoding="utf-8")

    trajectories = load_trajectory_batch([csv_path, json_path])

    assert np.allclose(trajectories[0], np.array([[0.0, 0.0], [1.0, 0.0]]))
    assert np.allclose(trajectories[1], np.array([[2.0, 0.0], [3.0, 0.0]]))


def test_load_trajectory_batch_rejects_invalid_format_and_extension(tmp_path) -> None:
    csv_path = tmp_path / "traj.csv"
    unknown_path = tmp_path / "traj.txt"

    with pytest.raises(ValueError, match="format must be"):
        load_trajectory_batch([csv_path], format="yaml")
    with pytest.raises(TrajectoryIOError, match="unsupported trajectory file extension"):
        load_trajectory_batch([unknown_path])


def test_load_trajectory_batch_wraps_parser_value_errors(tmp_path) -> None:
    csv_path = tmp_path / "traj.csv"
    csv_path.write_text("x\n0\n1\n", encoding="utf-8")

    with pytest.raises(TrajectoryIOError, match="could not load trajectory file"):
        load_trajectory_batch([csv_path], format="csv")

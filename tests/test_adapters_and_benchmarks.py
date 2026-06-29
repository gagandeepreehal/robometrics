from __future__ import annotations

import json

import pytest

from robometrics.adapters import (
    GenericCSVAdapter,
    GenericJSONAdapter,
    LeRobotStyleAdapter,
    MCAPAdapter,
    RLDSStyleAdapter,
    ROSStyleAdapter,
    get_adapter,
)
from robometrics.benchmarks import get_profile, list_profiles, run_profile


def test_generic_csv_adapter_preserves_z_column(tmp_path) -> None:
    path = tmp_path / "trajectory.csv"
    path.write_text("t,x,y,z\n0,0,0,1\n1,1,0,2\n", encoding="utf-8")
    adapter = GenericCSVAdapter()

    trajectory = adapter.load(path)
    report = adapter.validate(path)

    assert trajectory.points == [[0.0, 0.0, 1.0], [1.0, 0.0, 2.0]]
    assert report.passed is True
    assert report.dimensions == 3


def test_generic_json_adapter_loads_and_validates(tmp_path) -> None:
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps({"points": [[0.0, 0.0], [1.0, 0.0]]}), encoding="utf-8")
    adapter = GenericJSONAdapter()

    trajectory = adapter.load(path)
    report = adapter.validate(path)

    assert trajectory.points == [[0.0, 0.0], [1.0, 0.0]]
    assert report.passed is True
    assert adapter.metadata(path)["format"] == "json"


def test_ros_style_adapter_reads_pose_exports(tmp_path) -> None:
    path = tmp_path / "path.json"
    path.write_text(
        json.dumps(
            {
                "poses": [
                    {"pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.0}}},
                    {"pose": {"position": {"x": 1.0, "y": 0.0, "z": 0.0}}},
                ]
            }
        ),
        encoding="utf-8",
    )

    trajectory = ROSStyleAdapter().load(path)
    report = ROSStyleAdapter().validate(path)

    assert trajectory.points == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert report.passed is True
    assert report.row_count == 2
    assert report.dimensions == 3


def test_ros_style_adapter_reads_list_and_trajectory_exports(tmp_path) -> None:
    list_path = tmp_path / "list.json"
    list_path.write_text(json.dumps([[0.0, 0.0], [1.0, 1.0]]), encoding="utf-8")
    trajectory_path = tmp_path / "trajectory.json"
    trajectory_path.write_text(
        json.dumps({"trajectory": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]}),
        encoding="utf-8",
    )

    adapter = ROSStyleAdapter()

    assert adapter.load(list_path).points == [[0.0, 0.0], [1.0, 1.0]]
    assert adapter.load(trajectory_path).points == [[0.0, 0.0], [1.0, 1.0]]


def test_ros_style_adapter_validation_reports_adapter_errors(tmp_path) -> None:
    unsupported = tmp_path / "path.txt"
    unsupported.write_text("x,y\n0,0\n", encoding="utf-8")
    bad_json = tmp_path / "bad.json"
    bad_json.write_text(json.dumps({"trajectory": [{"x": 0.0}]}), encoding="utf-8")

    adapter = ROSStyleAdapter()

    assert adapter.validate(unsupported).issues[0].code == "unsupported_format"
    report = adapter.validate(bad_json)
    assert report.passed is False
    assert [issue.code for issue in report.issues] == [
        "missing_required_field",
        "adapter_error",
    ]


def test_lerobot_style_adapter_reads_steps(tmp_path) -> None:
    path = tmp_path / "episode.json"
    path.write_text(
        json.dumps({"steps": [{"observation": {"state": [0.0, 0.0]}}, {"state": [1.0, 0.0]}]}),
        encoding="utf-8",
    )

    trajectory = LeRobotStyleAdapter().load(path)
    report = LeRobotStyleAdapter().validate(path)

    assert trajectory.points == [[0.0, 0.0], [1.0, 0.0]]
    assert report.passed is True
    assert report.row_count == 2
    assert report.dimensions == 2


def test_lerobot_style_adapter_reads_directory_and_state_variants(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    path = dataset_dir / "data.json"
    path.write_text(
        json.dumps(
            {
                "states": [
                    {"x": 0.0, "y": 1.0, "z": 2.0},
                    {"position": [3.0, 4.0, 5.0]},
                    {"ee_position": [6.0, 7.0, 8.0]},
                ]
            }
        ),
        encoding="utf-8",
    )

    trajectory = LeRobotStyleAdapter().load(dataset_dir)

    assert trajectory.points == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]
    assert trajectory.metadata["path"] == str(path)


def test_lerobot_style_adapter_reads_step_lists_and_reports_errors(tmp_path) -> None:
    valid_list = tmp_path / "steps.json"
    valid_list.write_text(
        json.dumps([[0.0, 0.0], {"observation": {"state": [1.0, 1.0]}}]),
        encoding="utf-8",
    )
    invalid_payload = tmp_path / "invalid.json"
    invalid_payload.write_text(json.dumps({"metadata": {}}), encoding="utf-8")
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    adapter = LeRobotStyleAdapter()

    assert adapter.load(valid_list).points == [[0.0, 0.0], [1.0, 1.0]]
    assert adapter.validate(invalid_payload).issues[0].code == "adapter_error"
    assert adapter.validate(empty_dir).issues[0].code == "adapter_error"


def test_rlds_style_adapter_reads_episode_steps(tmp_path) -> None:
    path = tmp_path / "episode.json"
    path.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "steps": [
                            {"observation": {"position": [0.0, 0.0]}},
                            {"observation": {"position": [1.0, 0.0]}},
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    trajectory = RLDSStyleAdapter().load(path)
    report = RLDSStyleAdapter().validate(path)

    assert trajectory.points == [[0.0, 0.0], [1.0, 0.0]]
    assert report.passed is True
    assert report.row_count == 2
    assert report.dimensions == 2


def test_rlds_style_adapter_reads_trajectory_and_step_lists(tmp_path) -> None:
    trajectory_path = tmp_path / "trajectory.json"
    trajectory_path.write_text(
        json.dumps({"trajectory": [{"x": 0.0, "y": 1.0, "z": 2.0}]}),
        encoding="utf-8",
    )
    list_path = tmp_path / "steps.json"
    list_path.write_text(
        json.dumps(
            [
                {"observation": {"state": [0.0, 0.0]}},
                {"observation": {"ee_position": [1.0, 1.0]}},
                [2.0, 2.0],
            ]
        ),
        encoding="utf-8",
    )

    adapter = RLDSStyleAdapter()

    assert adapter.load(trajectory_path).points == [[0.0, 1.0, 2.0]]
    assert adapter.load(list_path).points == [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]


def test_rlds_style_adapter_validation_reports_errors(tmp_path) -> None:
    empty_episodes = tmp_path / "empty.json"
    empty_episodes.write_text(json.dumps({"episodes": []}), encoding="utf-8")
    invalid_record = tmp_path / "invalid.json"
    invalid_record.write_text(
        json.dumps({"steps": [{"observation": {"bad": 1.0}}]}),
        encoding="utf-8",
    )

    adapter = RLDSStyleAdapter()

    assert adapter.validate(empty_episodes).issues[0].code == "adapter_error"
    report = adapter.validate(invalid_record)
    assert report.passed is False
    assert "observation records require x/y" in report.issues[0].message


def test_mcap_adapter_explains_optional_dependency(tmp_path) -> None:
    adapter = MCAPAdapter()

    with pytest.raises(ImportError, match="MCAP loading is not part"):
        adapter.load(tmp_path / "run.mcap")

    report = adapter.validate(tmp_path / "run.mcap")
    assert report.issues[0].code == "optional_dependency_missing"


def test_get_adapter_dispatches_and_rejects_unknown() -> None:
    assert isinstance(get_adapter("generic-json"), GenericJSONAdapter)
    with pytest.raises(ValueError, match="unknown adapter"):
        get_adapter("unknown")


def test_benchmark_profiles_run_against_matching_trajectories() -> None:
    names = {profile.name for profile in list_profiles()}
    assert "policy_regression_ci" in names
    assert get_profile("trajectory_prediction_basic").metrics == (
        "ade",
        "fde",
        "hausdorff_distance",
    )

    result = run_profile(
        "policy_regression_ci",
        prediction=[[0.0, 0.0], [1.0, 0.0]],
        ground_truth=[[0.0, 0.0], [1.0, 0.0]],
    )

    assert result.strict_passed is True
    assert result.metadata["benchmark_profile"]["name"] == "policy_regression_ci"

from __future__ import annotations

import json

import pytest

from robometrics.adapters import (
    GenericJSONAdapter,
    LeRobotStyleAdapter,
    MCAPAdapter,
    RLDSStyleAdapter,
    ROSStyleAdapter,
    get_adapter,
)
from robometrics.benchmarks import get_profile, list_profiles, run_profile


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

    assert trajectory.points == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]


def test_lerobot_style_adapter_reads_steps(tmp_path) -> None:
    path = tmp_path / "episode.json"
    path.write_text(
        json.dumps({"steps": [{"observation": {"state": [0.0, 0.0]}}, {"state": [1.0, 0.0]}]}),
        encoding="utf-8",
    )

    trajectory = LeRobotStyleAdapter().load(path)

    assert trajectory.points == [[0.0, 0.0], [1.0, 0.0]]


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

    assert trajectory.points == [[0.0, 0.0], [1.0, 0.0]]


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

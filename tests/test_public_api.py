from __future__ import annotations

from pathlib import Path

import robometrics


def test_import_robometrics_exposes_version() -> None:
    assert isinstance(robometrics.__version__, str)
    assert robometrics.__version__


def test_public_api_exports_core_objects() -> None:
    expected = {
        "ade",
        "CheckpointEntry",
        "ComparisonResult",
        "Evaluator",
        "EvaluationHistory",
        "EvaluationResult",
        "EVALUATION_RESULT_SCHEMA_VERSION",
        "fde",
        "MetricComparison",
        "MetricResult",
        "MetricRegistry",
        "TrajectoryIOError",
        "UnknownMetricError",
        "acceleration",
        "acceleration_magnitude",
        "acceleration_limits_violated",
        "action_jerk",
        "average_displacement_error",
        "behavioral_diversity",
        "calibration_error",
        "collision_rate",
        "collision_rate_obb",
        "compounding_error_index",
        "contact_richness",
        "control_smoothness",
        "coverage_score",
        "curvature",
        "curvature_profile",
        "curvature_limits_violated",
        "dynamic_feasibility",
        "dynamic_feasibility_score",
        "displacement_at_k",
        "end_effector_tracking_error",
        "failure_severity",
        "final_displacement_error",
        "force_limit_compliance",
        "goal_reaching_accuracy",
        "grasp_success_rate",
        "intervention_free_time",
        "jerk",
        "jerk_cost",
        "jerk_magnitude",
        "jerk_limits_violated",
        "joint_limit_violation_rate",
        "kinematic_feasibility",
        "lane_departure_rate",
        "lateral_error",
        "load_pack",
        "load_trajectory_csv",
        "load_trajectory_dir",
        "load_trajectory_json",
        "long_horizon_drift",
        "longitudinal_error",
        "max_acceleration",
        "max_deceleration",
        "mean_acceleration",
        "mean_curvature",
        "min_ade",
        "min_distance_to_actors",
        "min_fde",
        "miss_rate",
        "near_miss_rate",
        "offroad_rate",
        "path_length",
        "physics_violation_rate",
        "prediction_nll",
        "recovery_success_rate",
        "registry",
        "rms_acceleration",
        "speed_profile",
        "soft_ttc",
        "task_success_rate",
        "temporal_drift",
        "time_to_collision",
        "topk_trajectory_error",
        "trajectory_diversity",
        "workspace_coverage",
    }

    assert expected.issubset(set(robometrics.__all__))
    for name in expected:
        assert hasattr(robometrics, name)


def test_public_api_reexports_trajectory_loaders() -> None:
    from robometrics import load_trajectory_csv, load_trajectory_dir, load_trajectory_json

    assert load_trajectory_csv is robometrics.load_trajectory_csv
    assert load_trajectory_dir is robometrics.load_trajectory_dir
    assert load_trajectory_json is robometrics.load_trajectory_json


def test_project_urls_use_canonical_repository() -> None:
    payload = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'Homepage = "https://github.com/gagandeepreehal/robometrics"' in payload
    assert "robometrics/robometrics" not in payload

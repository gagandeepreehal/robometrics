from __future__ import annotations

from pathlib import Path

import robometrics


def test_import_robometrics_exposes_version() -> None:
    assert isinstance(robometrics.__version__, str)
    assert robometrics.__version__


def test_public_api_exports_core_objects() -> None:
    expected = {
        "ade",
        "ComparisonResult",
        "Evaluator",
        "EvaluationResult",
        "fde",
        "MetricComparison",
        "MetricResult",
        "MetricRegistry",
        "TrajectoryIOError",
        "UnknownMetricError",
        "acceleration",
        "acceleration_magnitude",
        "acceleration_limits_violated",
        "average_displacement_error",
        "calibration_error",
        "collision_rate",
        "compounding_error_index",
        "contact_richness",
        "curvature",
        "curvature_profile",
        "dynamic_feasibility_score",
        "displacement_at_k",
        "end_effector_tracking_error",
        "final_displacement_error",
        "force_limit_compliance",
        "goal_reaching_accuracy",
        "grasp_success_rate",
        "jerk",
        "jerk_cost",
        "jerk_magnitude",
        "jerk_limits_violated",
        "joint_limit_violation_rate",
        "load_pack",
        "max_acceleration",
        "max_deceleration",
        "mean_acceleration",
        "mean_curvature",
        "min_ade",
        "min_distance_to_actors",
        "min_fde",
        "miss_rate",
        "offroad_rate",
        "path_length",
        "prediction_nll",
        "registry",
        "rms_acceleration",
        "speed_profile",
        "soft_ttc",
        "task_success_rate",
        "trajectory_diversity",
        "workspace_coverage",
    }

    assert expected.issubset(set(robometrics.__all__))
    for name in expected:
        assert hasattr(robometrics, name)


def test_project_urls_use_canonical_repository() -> None:
    payload = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'Homepage = "https://github.com/gagandeepreehal/robometrics"' in payload
    assert "robometrics/robometrics" not in payload

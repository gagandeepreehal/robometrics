from __future__ import annotations

import robometrics


def test_import_robometrics_exposes_version() -> None:
    assert isinstance(robometrics.__version__, str)
    assert robometrics.__version__


def test_public_api_exports_core_objects() -> None:
    expected = {
        "ade",
        "Evaluator",
        "EvaluationResult",
        "fde",
        "MetricResult",
        "MetricRegistry",
        "TrajectoryIOError",
        "UnknownMetricError",
        "acceleration",
        "acceleration_limits_violated",
        "average_displacement_error",
        "collision_rate",
        "curvature",
        "dynamic_feasibility_score",
        "final_displacement_error",
        "jerk",
        "jerk_cost",
        "jerk_limits_violated",
        "min_ade",
        "min_distance_to_actors",
        "min_fde",
        "miss_rate",
        "path_length",
        "registry",
        "speed_profile",
    }

    assert expected.issubset(set(robometrics.__all__))
    for name in expected:
        assert hasattr(robometrics, name)

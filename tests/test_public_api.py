from __future__ import annotations

import robometrics


def test_import_robometrics_exposes_version() -> None:
    assert isinstance(robometrics.__version__, str)
    assert robometrics.__version__


def test_public_api_exports_core_objects() -> None:
    expected = {
        "Evaluator",
        "EvaluationResult",
        "MetricResult",
        "MetricRegistry",
        "TrajectoryIOError",
        "UnknownMetricError",
        "average_displacement_error",
        "final_displacement_error",
        "registry",
    }

    assert expected.issubset(set(robometrics.__all__))
    for name in expected:
        assert hasattr(robometrics, name)

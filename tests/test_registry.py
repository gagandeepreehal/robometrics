from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from robometrics.registry import MetricRegistry, UnknownMetricError, registry
from robometrics.trajectory import average_displacement_error


def test_default_registry_lists_built_in_metrics() -> None:
    names = {metric.name for metric in registry.list_metrics()}

    assert "ade" in names
    assert "fde" in names
    assert "miss_rate" in names
    assert "temporal_drift" in names
    assert "coverage_score" in names
    assert "calibration_error" in names
    assert "dynamic_feasibility_score" in names
    assert "physics_violation_rate" in names
    assert "behavioral_diversity" in names
    assert registry.get("smoothness_score").required_inputs == ("traj",)
    assert registry.get("temporal_drift").required_inputs == ("predicted", "reference")


def test_registry_get_supports_aliases() -> None:
    assert registry.get("ade").fn is average_displacement_error
    assert registry.get("average_displacement_error").name == "ade"
    assert registry.get("ADE").name == "ade"
    assert registry.get("minade").name == "min_ade"
    assert registry.get("minfde").name == "min_fde"


def test_registry_filters_by_category() -> None:
    trajectory_metrics = registry.list_metrics(category="trajectory")

    assert trajectory_metrics
    assert {metric.category for metric in trajectory_metrics} == {"trajectory"}


def test_registry_rejects_unknown_metric() -> None:
    with pytest.raises(UnknownMetricError):
        registry.get("not_a_metric")


def test_custom_registry_registers_metric_metadata() -> None:
    custom = MetricRegistry()

    def metric() -> float:
        return 1.0

    registered = custom.register(
        name="example",
        fn=metric,
        category="custom",
        unit="score",
        description="Example metric.",
    )

    assert custom.get("example") == registered
    assert registered.description == "Example metric."


def test_custom_registry_registers_metrics_from_threads() -> None:
    custom = MetricRegistry()

    def register_metric(index: int) -> str:
        def metric() -> float:
            return float(index)

        return custom.register(
            name=f"example_{index}",
            fn=metric,
            category="custom",
        ).name

    with ThreadPoolExecutor(max_workers=4) as executor:
        names = list(executor.map(register_metric, range(8)))

    assert names == [f"example_{index}" for index in range(8)]
    assert {metric.name for metric in custom.list_metrics()} == set(names)

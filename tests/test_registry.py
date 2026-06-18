from __future__ import annotations

import pytest

from robometrics.registry import MetricRegistry, UnknownMetricError, registry
from robometrics.trajectory import average_displacement_error


def test_default_registry_lists_built_in_metrics() -> None:
    names = {metric.name for metric in registry.list_metrics()}

    assert "ade" in names
    assert "fde" in names
    assert "miss_rate" in names
    assert "dynamic_feasibility_score" in names


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

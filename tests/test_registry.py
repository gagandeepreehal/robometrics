from __future__ import annotations

import sys
import types
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress

import pytest

from robometrics.registry import MetricRegistry, UnknownMetricError, load_pack, registry
from robometrics.trajectory import average_displacement_error


def test_default_registry_lists_built_in_metrics() -> None:
    names = {metric.name for metric in registry.list_metrics()}

    assert "ade" in names
    assert "fde" in names
    assert "miss_rate" in names
    assert "dynamic_feasibility_score" in names
    assert registry.get("smoothness_score").required_inputs == ("traj",)


def test_default_registry_metrics_have_references() -> None:
    assert all(metric.reference for metric in registry.list_metrics())


def test_default_registry_records_metric_directionality() -> None:
    assert registry.get("task_success_rate").higher_is_better is True
    assert registry.get("time_to_collision").higher_is_better is True
    assert registry.get("min_distance_to_actors").higher_is_better is True
    assert registry.get("soft_ttc").higher_is_better is True
    assert registry.get("workspace_coverage").higher_is_better is True
    assert registry.get("contact_richness").higher_is_better is True
    assert registry.get("force_limit_compliance").higher_is_better is True
    assert registry.get("offroad_rate").higher_is_better is False
    assert registry.get("joint_limit_violation_rate").higher_is_better is False


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
    assert registered.higher_is_better is False


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


def test_register_many_registers_multiple_metrics() -> None:
    custom = MetricRegistry()

    def first() -> float:
        return 1.0

    def second() -> float:
        return 2.0

    registered = custom.register_many(
        [
            {
                "name": "first_metric",
                "fn": first,
                "category": "custom",
                "higher_is_better": True,
            },
            {"name": "second_metric", "fn": second, "category": "custom", "unit": "score"},
        ]
    )

    assert [metric.name for metric in registered] == ["first_metric", "second_metric"]
    assert custom.get("first_metric").higher_is_better is True
    assert custom.get("second_metric").unit == "score"


def test_register_many_missing_required_key_names_key() -> None:
    custom = MetricRegistry()

    with pytest.raises(ValueError, match="fn"):
        custom.register_many([{"name": "missing_fn", "category": "custom"}])


def test_unregister_removes_metric_and_aliases() -> None:
    custom = MetricRegistry()

    def metric() -> float:
        return 1.0

    custom.register(name="example", fn=metric, category="custom", aliases=("alias",))

    custom.unregister("alias")

    with pytest.raises(UnknownMetricError):
        custom.get("example")
    with pytest.raises(UnknownMetricError):
        custom.get("alias")


def test_load_pack_registers_in_memory_module() -> None:
    module_name = "robometrics_test_pack"
    module = types.ModuleType(module_name)

    def metric() -> float:
        return 1.0

    module.METRIC_PACK = [
        {
            "name": "test_pack_metric",
            "fn": metric,
            "category": "custom",
            "unit": "score",
            "reference": "Test Pack, 2026",
        }
    ]
    sys.modules[module_name] = module

    try:
        registered = load_pack(module_name)

        assert [metric.name for metric in registered] == ["test_pack_metric"]
        assert registry.get("test_pack_metric").reference == "Test Pack, 2026"
    finally:
        sys.modules.pop(module_name, None)
        with suppress(UnknownMetricError):
            registry.unregister("test_pack_metric")


def test_load_pack_can_register_into_custom_registry() -> None:
    module_name = "robometrics_custom_registry_pack"
    module = types.ModuleType(module_name)
    custom = MetricRegistry()

    def metric() -> float:
        return 1.0

    module.METRIC_PACK = [
        {
            "name": "isolated_pack_metric",
            "fn": metric,
            "category": "custom",
            "higher_is_better": True,
            "reference": "Test Pack, 2026",
        }
    ]
    sys.modules[module_name] = module

    try:
        registered = load_pack(module_name, registry=custom)

        assert [metric.name for metric in registered] == ["isolated_pack_metric"]
        assert custom.get("isolated_pack_metric").higher_is_better is True
        with pytest.raises(UnknownMetricError):
            registry.get("isolated_pack_metric")
    finally:
        sys.modules.pop(module_name, None)


def test_load_pack_missing_metric_pack_raises_value_error() -> None:
    module_name = "robometrics_missing_pack"
    sys.modules[module_name] = types.ModuleType(module_name)

    try:
        with pytest.raises(ValueError, match="METRIC_PACK"):
            load_pack(module_name)
    finally:
        sys.modules.pop(module_name, None)


def test_load_pack_missing_module_raises_import_error() -> None:
    with pytest.raises(ImportError):
        load_pack("robometrics_pack_that_does_not_exist")

from __future__ import annotations

import threading

import pytest

from robometrics import EvaluationResult, MetricAccumulator, MetricResult


def _result(*metrics: MetricResult) -> EvaluationResult:
    return EvaluationResult(results=list(metrics))


def _metric(name: str, value: float, *, unit: str = "meters") -> MetricResult:
    return MetricResult(
        name=name,
        value=value,
        unit=unit,
        metadata={
            "category": "trajectory",
            "description": f"{name} metric",
            "reference": "test reference",
            "is_novel": False,
        },
    )


def test_accumulator_identical_results_mean_equals_value() -> None:
    acc = MetricAccumulator(metrics=["ade", "fde"])

    acc.update(_result(_metric("ade", 1.0), _metric("fde", 2.0)))
    acc.update(_result(_metric("ade", 1.0), _metric("fde", 2.0)))
    summary = acc.compute()

    values = {metric.name: metric.value for metric in summary.results}
    assert values == {"ade": 1.0, "fde": 2.0}
    assert summary.metadata["accumulator"] is True
    assert summary.metadata["sample_count"] == 2


def test_accumulator_varying_values_aggregate_statistics() -> None:
    acc = MetricAccumulator(metrics=["ade"])

    for value in [1.0, 2.0, 3.0]:
        acc.update(_result(_metric("ade", value)))

    metric = acc.compute().results[0]
    assert metric.value == pytest.approx(2.0)
    assert metric.metadata["mean"] == pytest.approx(2.0)
    assert metric.metadata["std"] == pytest.approx(0.816496580927726)
    assert metric.metadata["min"] == 1.0
    assert metric.metadata["max"] == 3.0
    assert metric.metadata["finite_count"] == 3
    assert metric.metadata["sample_count"] == 3


def test_accumulator_reset_clears_state() -> None:
    acc = MetricAccumulator(metrics=["ade"])
    acc.update(_result(_metric("ade", 1.0)))

    acc.reset()

    assert acc.sample_count == 0
    with pytest.raises(RuntimeError):
        acc.compute()


def test_accumulator_sample_count_increments_and_resets() -> None:
    acc = MetricAccumulator()

    acc.update(_result(_metric("ade", 1.0)))
    acc.update(_result(_metric("ade", 2.0)))
    assert acc.sample_count == 2

    acc.reset()
    assert acc.sample_count == 0


def test_accumulator_metric_filter_tracks_only_specified_metrics() -> None:
    acc = MetricAccumulator(metrics=["ade"])

    acc.update(_result(_metric("ade", 1.0), _metric("fde", 2.0)))
    summary = acc.compute()

    assert [metric.name for metric in summary.results] == ["ade"]
    assert acc.tracked_metrics == ["ade"]


def test_accumulator_tracks_all_seen_metric_names_without_filter() -> None:
    acc = MetricAccumulator()

    acc.update(_result(_metric("ade", 1.0)))
    acc.update(_result(_metric("fde", 2.0)))

    assert acc.tracked_metrics == ["ade", "fde"]


def test_accumulator_concurrent_updates_do_not_corrupt_state() -> None:
    acc = MetricAccumulator(metrics=["ade"])

    def update_many() -> None:
        for _ in range(100):
            acc.update(_result(_metric("ade", 1.0)))

    threads = [threading.Thread(target=update_many), threading.Thread(target=update_many)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    summary = acc.compute()
    assert acc.sample_count == 200
    assert summary.results[0].metadata["finite_count"] == 200
    assert summary.results[0].value == 1.0


def test_accumulator_compute_before_update_raises() -> None:
    with pytest.raises(RuntimeError):
        MetricAccumulator().compute()

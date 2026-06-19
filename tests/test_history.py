from __future__ import annotations

import math

import pytest

from robometrics import EvaluationHistory, EvaluationResult, MetricResult


def _result(**values: float) -> EvaluationResult:
    return EvaluationResult(
        results=[
            MetricResult(name=name, value=value, metadata={"category": "trajectory"})
            for name, value in values.items()
        ]
    )


def test_history_record_and_metric_values() -> None:
    history = EvaluationHistory()

    history.record(step=10, result=_result(ade=1.0), label="first")
    history.record(step=5, result=_result(ade=2.0), label="zero")

    assert history.steps() == [5, 10]
    assert history.metric_values("ade") == {5: 2.0, 10: 1.0}
    assert history.get(10).results[0].value == 1.0


def test_history_duplicate_step_raises() -> None:
    history = EvaluationHistory()
    history.record(step=1, result=_result(ade=1.0))

    with pytest.raises(ValueError, match="already recorded"):
        history.record(step=1, result=_result(ade=2.0))


def test_history_trend_linearly_increasing_metric() -> None:
    history = EvaluationHistory()
    history.record(step=0, result=_result(ade=1.0))
    history.record(step=10, result=_result(ade=3.0))
    history.record(step=20, result=_result(ade=5.0))

    assert history.trend("ade") == pytest.approx(0.2)


def test_history_trend_with_fewer_than_two_finite_values_is_nan() -> None:
    history = EvaluationHistory()
    history.record(step=0, result=_result(ade=1.0))
    history.record(step=10, result=_result(fde=3.0))

    assert math.isnan(history.trend("ade"))


def test_history_best_step_defaults_to_lower_is_better() -> None:
    history = EvaluationHistory()
    history.record(step=0, result=_result(ade=2.0))
    history.record(step=10, result=_result(ade=1.0))

    assert history.best_step("ade") == 10


def test_history_best_step_higher_is_better() -> None:
    history = EvaluationHistory()
    history.record(step=0, result=_result(task_success_rate=0.5))
    history.record(step=10, result=_result(task_success_rate=0.8))

    assert history.best_step("task_success_rate", higher_is_better=True) == 10


def test_history_best_step_with_no_finite_values_raises() -> None:
    history = EvaluationHistory()
    history.record(
        step=0,
        result=EvaluationResult(
            results=[
                MetricResult(
                    name="ade",
                    value=float("nan"),
                    metadata={"error": "failed"},
                )
            ]
        ),
    )

    with pytest.raises(ValueError, match="no finite values"):
        history.best_step("ade")


def test_history_json_round_trip_preserves_metric_values() -> None:
    history = EvaluationHistory()
    history.record(step=0, result=_result(ade=2.0), label="initial")
    history.record(step=10, result=_result(ade=1.0), label="later")

    restored = EvaluationHistory.from_json(history.to_json())

    assert restored.metric_values("ade") == history.metric_values("ade")
    assert restored.steps() == [0, 10]


def test_history_to_markdown_contains_steps_and_metrics() -> None:
    history = EvaluationHistory()
    history.record(step=0, result=_result(ade=2.0), label="initial")
    history.record(step=10, result=_result(fde=1.0))

    markdown = history.to_markdown()

    assert "Step" in markdown
    assert "ade" in markdown
    assert "fde" in markdown
    assert "| 0 | initial |" in markdown
    assert "| 10 | - |" in markdown

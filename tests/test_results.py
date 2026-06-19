from __future__ import annotations

import json

import pandas as pd
import pytest

from robometrics.results import EvaluationResult, MetricResult


def test_metric_result_exports_dict_and_json() -> None:
    result = MetricResult(
        name="ade",
        value=0.42,
        unit="meters",
        passed=True,
        threshold=1.0,
        metadata={"category": "trajectory"},
    )

    assert result.to_dict()["name"] == "ade"
    assert json.loads(result.to_json())["value"] == 0.42


def test_metric_result_json_replaces_non_finite_values_with_null() -> None:
    result = MetricResult(
        name="miss_rate",
        value=float("nan"),
        metadata={"raw_value": [1.0, float("inf")]},
    )

    payload = result.to_dict()
    assert payload["value"] is None
    assert payload["metadata"]["value_serialization"]["original"] == "nan"
    assert payload["metadata"]["raw_value"] == [1.0, None]
    assert json.loads(result.to_json())["value"] is None


def test_metric_result_json_marks_infinite_values() -> None:
    result = MetricResult(name="min_distance_to_actors", value=float("inf"), unit="meters")

    payload = result.to_dict()

    assert payload["value"] is None
    assert payload["metadata"]["value_serialization"] == {
        "original": "inf",
        "json_value": None,
    }


def test_evaluation_result_summary_and_exports() -> None:
    result = EvaluationResult(
        results=[
            MetricResult(
                name="ade",
                value=0.42,
                unit="meters",
                passed=True,
                threshold=1.0,
                metadata={"category": "trajectory"},
            ),
            MetricResult(
                name="fde",
                value=0.71,
                unit="meters",
                passed=None,
                metadata={"category": "trajectory"},
            ),
        ]
    )

    summary = result.summary()
    assert summary["metric_count"] == 2
    assert summary["categories"] == ["trajectory"]
    assert summary["aggregate"]["mean"] == pytest.approx(0.565)
    assert "warning" not in summary["aggregate"]
    assert summary["per_unit"]["meters"]["mean"] == pytest.approx(0.565)

    markdown = result.to_markdown()
    assert "| Metric | Value | Unit | Passed | Threshold |" in markdown
    assert "ADE" in markdown

    frame = result.to_dataframe()
    assert isinstance(frame, pd.DataFrame)
    assert list(frame["name"]) == ["ade", "fde"]


def test_evaluation_result_accepts_metrics_alias() -> None:
    metric = MetricResult(name="example", value=1.0)
    result = EvaluationResult(metrics=[metric])

    assert result.results == [metric]
    assert result.metrics == [metric]


def test_evaluation_result_summary_warns_for_mixed_units() -> None:
    result = EvaluationResult(
        results=[
            MetricResult(name="ade", value=1.0, unit="meters"),
            MetricResult(name="jerk_cost", value=10.0, unit="m^2/s^6"),
        ]
    )

    summary = result.summary()

    assert summary["aggregate"]["warning"] == "aggregate mixes metric units; use per_unit summaries"
    assert set(summary["per_unit"]) == {"m^2/s^6", "meters"}

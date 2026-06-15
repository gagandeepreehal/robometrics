from __future__ import annotations

import json

import pandas as pd
import pytest

from robotmetrics.results import EvaluationResult, MetricResult


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

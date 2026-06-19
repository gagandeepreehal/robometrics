from __future__ import annotations

import json

import numpy as np
import pytest

import robometrics.cli as cli
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


def test_metric_result_round_trips_non_finite_json() -> None:
    nan_result = MetricResult.from_json(MetricResult(name="nan", value=float("nan")).to_json())
    neg_inf_result = MetricResult.from_json(
        MetricResult(name="neg_inf", value=float("-inf")).to_json()
    )

    assert nan_result.value != nan_result.value
    assert neg_inf_result.value == float("-inf")

    with pytest.raises(ValueError, match="MetricResult JSON"):
        MetricResult.from_json("[]")


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
    assert "per_unit" not in summary
    assert result.strict_passed is True

    markdown = result.to_markdown()
    assert "| Metric | Value | Unit | Passed | Threshold |" in markdown
    assert "ADE" in markdown

    pd = pytest.importorskip("pandas")
    frame = result.to_dataframe()
    assert isinstance(frame, pd.DataFrame)
    assert list(frame["name"]) == ["ade", "fde"]


def test_evaluation_result_exports_csv(tmp_path) -> None:
    pytest.importorskip("pandas")
    result = EvaluationResult(results=[MetricResult(name="ade", value=0.42, unit="meters")])
    path = tmp_path / "results.csv"

    csv_text = result.to_csv(path)

    assert "name,value,unit" in csv_text
    assert "ade,0.42,meters" in csv_text
    assert path.read_text(encoding="utf-8") == csv_text


def test_evaluation_result_accepts_metrics_alias() -> None:
    metric = MetricResult(name="example", value=1.0)
    result = EvaluationResult(metrics=[metric])

    assert result.results == [metric]
    assert result.metrics == [metric]

    replacement = [MetricResult(name="replacement", value=2.0)]
    result.metrics = replacement
    assert result.results == replacement

    with pytest.raises(ValueError, match="provide either"):
        EvaluationResult(results=[metric], metrics=[metric])


def test_evaluation_result_round_trips_json() -> None:
    result = EvaluationResult(
        results=[
            MetricResult(
                name="min_distance_to_actors",
                value=float("inf"),
                unit="meters",
                metadata={"category": "safety"},
            )
        ],
        metadata={"robometrics_version": "test"},
    )

    restored = EvaluationResult.from_json(result.to_json())

    assert restored.metadata == {"robometrics_version": "test"}
    assert restored.results[0].name == "min_distance_to_actors"
    assert restored.results[0].value == float("inf")
    assert restored.to_dict()["results"][0]["value"] is None

    with pytest.raises(ValueError, match="results list"):
        EvaluationResult.from_dict({})
    with pytest.raises(ValueError, match="metadata"):
        EvaluationResult.from_dict({"results": [], "metadata": []})
    with pytest.raises(ValueError, match="results must contain objects"):
        EvaluationResult.from_dict({"results": [1]})
    with pytest.raises(ValueError, match="EvaluationResult JSON"):
        EvaluationResult.from_json("[]")


def test_evaluation_result_strict_passed_ignores_unthresholded_metrics() -> None:
    result = EvaluationResult(
        results=[
            MetricResult(name="ade", value=0.5, passed=True),
            MetricResult(name="fde", value=0.1, passed=None),
            MetricResult(name="jerk", value=99.0, passed=False),
        ]
    )

    assert result.passed is None
    assert result.strict_passed is False


def test_evaluation_result_summary_warns_for_mixed_units() -> None:
    result = EvaluationResult(
        results=[
            MetricResult(name="ade", value=1.0, unit="meters"),
            MetricResult(name="jerk_cost", value=10.0, unit="m^2/s^6"),
        ]
    )

    summary = result.summary()

    assert summary["aggregate"]["warning"] == "aggregate mixes metric units; use per_unit summaries"
    assert summary["aggregate"]["mean"] is None
    assert set(summary["per_unit"]) == {"m^2/s^6", "meters"}


def test_evaluation_result_empty_and_nonfinite_formatting() -> None:
    empty = EvaluationResult()
    assert empty.passed is None
    assert empty.strict_passed is None
    assert empty.summary()["aggregate"]["mean"] is None

    result = EvaluationResult(
        results=[
            MetricResult(name="ade", value=float("nan")),
            MetricResult(name="fde", value=float("inf")),
            MetricResult(name="min_ade", value=float("-inf")),
        ],
        metadata={"array": np.array([1]), "scalar": np.float64(2)},
    )
    markdown = result.to_markdown()
    payload = result.to_dict()

    assert "nan" in markdown
    assert "inf" in markdown
    assert "-inf" in markdown
    assert payload["metadata"] == {"array": [1], "scalar": 2}


def test_comparison_identical_results_are_ties() -> None:
    result = EvaluationResult(
        results=[
            MetricResult(name="ade", value=1.0),
            MetricResult(name="task_success_rate", value=0.8),
        ]
    )

    comparison = result.compare(result)

    assert all(item.winner == "tie" for item in comparison.comparisons)
    assert all(item.delta == 0.0 for item in comparison.comparisons)
    assert comparison.winner_count() == {"a": 0, "b": 0, "tie": 2}


def test_comparison_a_wins_all_metrics() -> None:
    result_a = EvaluationResult(
        results=[
            MetricResult(name="ade", value=1.0),
            MetricResult(name="task_success_rate", value=0.9),
            MetricResult(name="collision_rate", value=0.0),
        ]
    )
    result_b = EvaluationResult(
        results=[
            MetricResult(name="ade", value=2.0),
            MetricResult(name="task_success_rate", value=0.5),
            MetricResult(name="collision_rate", value=0.25),
        ]
    )

    comparison = result_a.compare(result_b)

    assert comparison.winner_count()["a"] == 3
    assert {item.name: item.higher_is_better for item in comparison.comparisons} == {
        "ade": False,
        "task_success_rate": True,
        "collision_rate": False,
    }


def test_comparison_markdown_contains_metric_names() -> None:
    result_a = EvaluationResult(
        results=[
            MetricResult(name="ade", value=1.0),
            MetricResult(name="task_success_rate", value=0.9),
        ]
    )
    result_b = EvaluationResult(
        results=[
            MetricResult(name="ade", value=2.0),
            MetricResult(name="task_success_rate", value=0.5),
        ]
    )

    markdown = result_a.compare(result_b).to_markdown()

    assert "ade" in markdown
    assert "task_success_rate" in markdown


def test_cli_compare_outputs_text(tmp_path, capsys) -> None:
    result_a = EvaluationResult(results=[MetricResult(name="ade", value=1.0)])
    result_b = EvaluationResult(results=[MetricResult(name="ade", value=2.0)])
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    a_path.write_text(result_a.to_json(), encoding="utf-8")
    b_path.write_text(result_b.to_json(), encoding="utf-8")

    assert cli.main(["compare", str(a_path), str(b_path), "--format", "text"]) == 0

    assert capsys.readouterr().out.strip()

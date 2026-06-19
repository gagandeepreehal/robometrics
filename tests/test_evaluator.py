from __future__ import annotations

import numpy as np
import pytest

from robometrics import EvaluationInputError, Evaluator, UnknownMetricError, __version__


def test_evaluator_runs_named_metrics_with_thresholds() -> None:
    pred = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    gt = np.array([[0.0, 0.0], [1.1, 0.0], [2.1, 0.0]])

    result = Evaluator().evaluate(
        prediction=pred,
        ground_truth=gt,
        metrics=["ade", "fde"],
        thresholds={"ade": 1.0, "fde": 1.0},
    )

    assert [metric.name for metric in result.results] == ["ade", "fde"]
    assert result.results[0].value == pytest.approx((0.0 + 0.1 + 0.1) / 3.0)
    assert result.results[0].passed is True
    assert result.passed is True
    assert result.metadata["robometrics_version"] == __version__


def test_evaluator_supports_category_selection() -> None:
    pred = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    gt = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    result = Evaluator().evaluate(
        prediction=pred,
        ground_truth=gt,
        categories=["trajectory"],
    )

    assert {metric.metadata["category"] for metric in result.results} == {"trajectory"}
    assert {"ade", "fde", "hausdorff_distance"}.issubset(
        {metric.name for metric in result.results}
    )


def test_evaluator_all_runs_only_compatible_metrics() -> None:
    pred = np.array([[0.0, 0.0], [1.0, 0.0]])
    gt = np.array([[0.0, 0.0], [1.0, 0.0]])

    result = Evaluator().evaluate(prediction=pred, ground_truth=gt, metrics="all")
    names = {metric.name for metric in result.results}

    assert "ade" in names
    assert "min_ade" not in names
    assert result.metadata["skipped_metrics"]


def test_evaluator_runs_prediction_metrics_with_metric_kwargs() -> None:
    gt = np.array([[0.0, 0.0], [1.0, 0.0]])
    predictions = np.array(
        [
            [[0.0, 0.0], [2.0, 0.0]],
            [[0.0, 0.0], [1.2, 0.0]],
        ]
    )

    result = Evaluator().evaluate(
        prediction=predictions,
        ground_truth=gt,
        metrics=["min_ade", "miss_rate", "topk_trajectory_error"],
        metric_kwargs={"topk_trajectory_error": {"k": 2}},
        threshold=0.5,
    )

    values = {metric.name: metric.value for metric in result.results}
    assert values["min_ade"] == pytest.approx(0.1)
    assert values["miss_rate"] == 0.0
    assert values["topk_trajectory_error"] == pytest.approx(0.1)


def test_evaluator_stores_metric_failures() -> None:
    pred = np.array([[0.0, 0.0], [1.0, 0.0]])
    gt = np.array([[0.0, 0.0], [1.0, 0.0]])

    result = Evaluator().evaluate(
        prediction=pred,
        ground_truth=gt,
        metrics=["ade", "min_ade"],
    )

    failure = result.results[1]
    assert failure.name == "min_ade"
    assert failure.passed is False
    assert "error" in failure.metadata


def test_evaluator_rejects_unknown_metric_and_category() -> None:
    with pytest.raises(UnknownMetricError):
        Evaluator().evaluate(prediction=np.array([[0.0, 0.0]]), metrics=["missing"])

    with pytest.raises(EvaluationInputError):
        Evaluator().evaluate(prediction=np.array([[0.0, 0.0]]), categories=["missing"])


def test_evaluator_requires_some_input() -> None:
    with pytest.raises(EvaluationInputError):
        Evaluator().evaluate(metrics=["ade"])


def test_evaluator_reports_missing_required_inputs_for_named_metric() -> None:
    pred = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    result = Evaluator().evaluate(prediction=pred, categories=["comfort"])
    assert {metric.name for metric in result.results} == {"smoothness_score"}

    explicit = Evaluator().evaluate(prediction=pred, metrics=["acceleration"])
    assert (
        explicit.results[0].metadata["error"]
        == "provided inputs are not compatible with this metric"
    )


def test_evaluator_evaluates_dataset() -> None:
    predictions = [
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([[0.0, 0.0], [1.2, 0.0]]),
    ]
    ground_truths = [
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([[0.0, 0.0], [1.0, 0.0]]),
    ]

    result = Evaluator().evaluate_dataset(
        predictions=predictions,
        ground_truths=ground_truths,
        metrics=["ade"],
        thresholds={"ade": 0.2},
    )

    assert result.metadata["dataset"] is True
    assert result.metadata["sample_count"] == 2
    assert result.results[0].name == "ade"
    assert result.results[0].value == pytest.approx(0.05)
    assert result.results[0].metadata["sample_count"] == 2
    assert result.strict_passed is True


def test_evaluator_rejects_bad_dataset_inputs() -> None:
    with pytest.raises(EvaluationInputError, match="same length"):
        Evaluator().evaluate_dataset(
            predictions=[np.array([[0.0, 0.0]])],
            ground_truths=[],
            metrics=["ade"],
        )


def test_evaluator_rejects_invalid_common_inputs() -> None:
    with pytest.raises(EvaluationInputError):
        Evaluator().evaluate(
            prediction=np.array([[0.0, np.nan]]),
            ground_truth=np.array([[0.0, 0.0]]),
            metrics=["ade"],
        )

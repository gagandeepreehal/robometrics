from __future__ import annotations

import numpy as np
import pytest

from robometrics import (
    EvaluationInputError,
    Evaluator,
    MetricRegistry,
    Trajectory,
    UnknownMetricError,
    __version__,
)


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
    assert result.results[0].metadata["reference"] == "Alahi et al., Social Force, CVPR 2016"
    assert result.results[0].metadata["is_novel"] is False
    assert result.metadata["robometrics_version"] == __version__


def test_evaluator_thresholds_use_metric_direction() -> None:
    custom = MetricRegistry()

    def score_metric() -> float:
        return 0.8

    custom.register(
        name="score_metric",
        fn=score_metric,
        category="custom",
        higher_is_better=True,
    )

    result = Evaluator(metric_registry=custom).evaluate(
        prediction=np.array([[0.0, 0.0], [1.0, 0.0]]),
        metrics=["score_metric"],
        thresholds={"score_metric": 0.7},
    )

    assert result.results[0].passed is True

    failing = Evaluator(metric_registry=custom).evaluate(
        prediction=np.array([[0.0, 0.0], [1.0, 0.0]]),
        metrics=["score_metric"],
        thresholds={"score_metric": 0.9},
    )

    assert failing.results[0].passed is False


def test_evaluator_accepts_trajectory_schema_inputs() -> None:
    pred = Trajectory(points=[[0.0, 0.0], [1.0, 0.0]], timestamps=[0.0, 0.1])
    gt = Trajectory(points=[[0.0, 0.0], [1.2, 0.0]], timestamps=[0.0, 0.1])

    result = Evaluator().evaluate(
        prediction=pred,
        ground_truth=gt,
        metrics=["ade", "fde"],
    )

    assert [metric.name for metric in result.results] == ["ade", "fde"]
    assert result.results[0].value == pytest.approx(0.1)
    assert result.results[1].value == pytest.approx(0.2)


def test_evaluator_accepts_trajectory_schema_metric_inputs() -> None:
    ego = Trajectory(points=[[0.0, 0.0], [1.0, 0.0]])
    actor = Trajectory(points=[[0.0, 2.0], [1.0, 0.2]])

    result = Evaluator().evaluate(
        ego_traj=ego,
        actor_trajs=[actor],
        ego_radius=0.5,
        actor_radius=0.5,
        metrics=["collision_rate"],
    )

    assert result.results[0].value == pytest.approx(0.5)


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


def test_evaluator_runs_prediction_nll_with_registered_gt_input() -> None:
    gt = np.array([[0.0, 0.0], [1.0, 0.0]])
    predictions = gt[None, :, :]

    result = Evaluator().evaluate(
        prediction=predictions,
        ground_truth=gt,
        log_weights=np.array([0.0]),
        metrics=["prediction_nll"],
    )

    assert result.results[0].value == pytest.approx(0.0)
    assert "error" not in result.results[0].metadata


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
    assert failure.passed is None
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


def test_evaluator_dataset_thresholds_use_metric_direction() -> None:
    custom = MetricRegistry()

    def score_metric() -> float:
        return 0.8

    custom.register(
        name="score_metric",
        fn=score_metric,
        category="custom",
        higher_is_better=True,
    )

    result = Evaluator(metric_registry=custom).evaluate_dataset(
        predictions=[np.array([[0.0, 0.0], [1.0, 0.0]])],
        ground_truths=[np.array([[0.0, 0.0], [1.0, 0.0]])],
        metrics=["score_metric"],
        thresholds={"score_metric": 0.7},
    )

    assert result.results[0].metadata["higher_is_better"] is True
    assert result.results[0].passed is True
    assert result.strict_passed is True


def test_evaluator_dataset_bootstrap_ci_contains_mean() -> None:
    predictions = [
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([[0.0, 0.0], [1.2, 0.0]]),
        np.array([[0.0, 0.0], [1.4, 0.0]]),
    ]
    ground_truths = [
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([[0.0, 0.0], [1.0, 0.0]]),
    ]

    result = Evaluator().evaluate_dataset(
        predictions=predictions,
        ground_truths=ground_truths,
        metrics=["ade"],
        bootstrap_ci=1000,
    )

    metric = result.results[0]
    assert metric.metadata["ci_lower"] <= metric.value <= metric.metadata["ci_upper"]
    assert metric.metadata["bootstrap_n"] == 1000
    assert metric.metadata["ci_alpha"] == 0.05


def test_evaluator_dataset_bootstrap_ci_varies_when_values_vary() -> None:
    predictions = [
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([[0.0, 0.0], [3.0, 0.0]]),
        np.array([[0.0, 0.0], [5.0, 0.0]]),
    ]
    ground_truths = [np.array([[0.0, 0.0], [1.0, 0.0]]) for _ in predictions]

    metric = Evaluator().evaluate_dataset(
        predictions=predictions,
        ground_truths=ground_truths,
        metrics=["ade"],
        bootstrap_ci=1000,
    ).results[0]

    assert metric.metadata["ci_lower"] < metric.metadata["ci_upper"]


def test_evaluator_dataset_bootstrap_seed_is_configurable() -> None:
    predictions = [
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([[0.0, 0.0], [3.0, 0.0]]),
        np.array([[0.0, 0.0], [5.0, 0.0]]),
    ]
    ground_truths = [np.array([[0.0, 0.0], [1.0, 0.0]]) for _ in predictions]
    kwargs = {
        "predictions": predictions,
        "ground_truths": ground_truths,
        "metrics": ["ade"],
        "bootstrap_ci": 1000,
    }

    first = Evaluator().evaluate_dataset(**kwargs, bootstrap_seed=1).results[0]
    second = Evaluator().evaluate_dataset(**kwargs, bootstrap_seed=1).results[0]
    third = Evaluator().evaluate_dataset(**kwargs, bootstrap_seed=2).results[0]

    assert first.metadata["bootstrap_seed"] == 1
    assert third.metadata["bootstrap_seed"] == 2
    assert first.metadata["ci_lower"] == second.metadata["ci_lower"]
    assert first.metadata["ci_upper"] == second.metadata["ci_upper"]


def test_evaluator_dataset_bootstrap_ci_degenerate_identical_values() -> None:
    predictions = [np.array([[0.0, 0.0], [2.0, 0.0]]) for _ in range(3)]
    ground_truths = [np.array([[0.0, 0.0], [1.0, 0.0]]) for _ in predictions]

    metric = Evaluator().evaluate_dataset(
        predictions=predictions,
        ground_truths=ground_truths,
        metrics=["ade"],
        bootstrap_ci=1000,
    ).results[0]

    assert metric.metadata["ci_lower"] == pytest.approx(metric.metadata["ci_upper"])
    assert metric.metadata["ci_lower"] == pytest.approx(metric.value)


def test_evaluator_rejects_too_small_bootstrap_ci() -> None:
    with pytest.raises(EvaluationInputError, match="bootstrap_ci must be at least 100"):
        Evaluator().evaluate_dataset(
            predictions=[np.array([[0.0, 0.0]])],
            ground_truths=[np.array([[0.0, 0.0]])],
            metrics=["ade"],
            bootstrap_ci=50,
        )


def test_evaluator_rejects_invalid_ci_alpha() -> None:
    with pytest.raises(EvaluationInputError, match="ci_alpha"):
        Evaluator().evaluate_dataset(
            predictions=[np.array([[0.0, 0.0]])],
            ground_truths=[np.array([[0.0, 0.0]])],
            metrics=["ade"],
            bootstrap_ci=100,
            ci_alpha=0.6,
        )


def test_evaluator_rejects_invalid_bootstrap_seed() -> None:
    with pytest.raises(EvaluationInputError, match="bootstrap_seed"):
        Evaluator().evaluate_dataset(
            predictions=[np.array([[0.0, 0.0]])],
            ground_truths=[np.array([[0.0, 0.0]])],
            metrics=["ade"],
            bootstrap_ci=100,
            bootstrap_seed=True,
        )


def test_evaluator_dataset_default_has_no_bootstrap_ci_metadata() -> None:
    metric = Evaluator().evaluate_dataset(
        predictions=[np.array([[0.0, 0.0], [1.0, 0.0]])],
        ground_truths=[np.array([[0.0, 0.0], [1.0, 0.0]])],
        metrics=["ade"],
    ).results[0]

    assert "ci_lower" not in metric.metadata
    assert "ci_upper" not in metric.metadata


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

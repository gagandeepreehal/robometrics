# Evaluator

This page is a discoverable entry point for the lightweight evaluator. The full guide is also available in [Evaluator](evaluation.md).
The same examples are available as a runnable script in `examples/evaluator_usage.py`.

## Single-Trajectory Metrics

```python
import numpy as np

from robometrics import Evaluator

pred = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
gt = np.array([[0.0, 0.0], [1.1, 0.0], [2.2, 0.0]])

trajectory_result = Evaluator().evaluate(
    prediction=pred,
    ground_truth=gt,
    metrics=["ade", "fde"],
    thresholds={"ade": 0.5, "fde": 1.0},
)

print(trajectory_result.summary())
print(trajectory_result.to_markdown())
```

## Multimodal Prediction Metrics

Prediction metrics such as `min_ade`, `min_fde`, `miss_rate`, and `topk_trajectory_error` expect a `KxTx2` or `KxTx3` prediction array and a `Tx2` or `Tx3` ground-truth trajectory.

```python
import numpy as np

from robometrics import Evaluator

predictions = np.array(
    [
        [[0.0, 0.0], [1.8, 0.0], [3.0, 0.0]],
        [[0.0, 0.0], [1.1, 0.0], [2.2, 0.0]],
    ]
)
gt = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

prediction_result = Evaluator().evaluate(
    prediction=predictions,
    ground_truth=gt,
    metrics=["min_ade", "min_fde", "miss_rate", "topk_trajectory_error"],
    threshold=0.5,
    thresholds={
        "min_ade": 0.5,
        "min_fde": 0.5,
        "miss_rate": 0.0,
        "topk_trajectory_error": 0.5,
    },
    metric_kwargs={"topk_trajectory_error": {"k": 2}},
)

print(prediction_result.summary())
print(prediction_result.to_json())
```

## Dataset Aggregation

Use `evaluate_dataset()` when each sample has one prediction and one
ground-truth trajectory:

```python
dataset_result = Evaluator().evaluate_dataset(
    predictions=[pred_a, pred_b],
    ground_truths=[gt_a, gt_b],
    metrics=["ade", "fde"],
)
```

The result contains one aggregate metric row per metric, with per-sample values
and min/max/std metadata.

## Threshold Direction

Evaluator thresholds follow registry directionality. Lower-is-better metrics
such as `ade`, `fde`, and collision rates pass when `value <= threshold`.
Higher-is-better metrics such as `task_success_rate`, `workspace_coverage`, and
`force_limit_compliance` pass when `value >= threshold`. The direction is stored
in result metadata as `higher_is_better`.

## Error Behavior

Unknown metric names raise `UnknownMetricError` before evaluation starts. If a known metric cannot run with the supplied inputs (for example, a metric that expects a `KxTx2` array is given a `Tx2` array), evaluation continues and that metric is returned as a `MetricResult` with `value=nan`, `passed=None`, and `metadata["error"]` describing the failure. The error is counted in `summary()["error_count"]`.

This is intentional: errors are **explicit and visible** rather than silently skipped. Check `error_count` in CI or iterate over results to find metrics with `"error"` in their metadata.

For automatic category selection, if every selected metric is skipped because a
required input is missing, the evaluator reports the missing input names instead
of the generic "no compatible metrics" message.

Array-valued metric functions are reduced to scalar evaluator results. Vector
arrays use mean row-wise norm, scalar arrays use mean value, and the raw array
plus reduction name are stored in metadata.

Use `strict_passed` for CI gates that should consider only thresholded metrics.
Metrics that error at runtime have `passed=None` and are therefore **excluded**
from `strict_passed`, so a runtime error alone will not fail a CI gate that has
no threshold for that metric.

# Evaluator

RoboMetrics includes a small local evaluator for running named metrics together. It is optional; direct metric functions remain the main API.

## Evaluator

```python
from robometrics import Evaluator

evaluator = Evaluator()
result = evaluator.evaluate(
    prediction=pred,
    ground_truth=gt,
    metrics=["ade", "fde"],
)
```

`prediction` and `ground_truth` are mapped to the argument names used by built-in metrics. They may be NumPy-like arrays or `Trajectory` schema objects. Metric-specific inputs can be supplied as keyword arguments, such as `dt`, `actor_trajs`, `ego_radius`, `actor_radius`, `lane_boundary`, `constraints`, `k`, or `threshold`.
If an automatically selected category has no runnable metrics because a
required input is missing, the evaluator reports the missing input names, such
as `dt` for comfort metrics.

## Metric Selection

Run named metrics:

```python
result = evaluator.evaluate(
    prediction=pred,
    ground_truth=gt,
    metrics=["ade", "fde"],
)
```

Run every compatible registered metric for the provided inputs:

```python
result = evaluator.evaluate(
    prediction=pred,
    ground_truth=gt,
    metrics="all",
)
```

Run a category:

```python
result = evaluator.evaluate(
    prediction=pred,
    ground_truth=gt,
    categories=["trajectory"],
)
```

## Thresholds

Pass/fail thresholds use registry directionality. Lower-is-better metrics pass
with `value <= threshold`; metrics marked `higher_is_better=True` pass with
`value >= threshold`. The same rule is used for single-sample evaluation and
`evaluate_dataset()` aggregation.

```python
result = evaluator.evaluate(
    prediction=pred,
    ground_truth=gt,
    metrics=["ade", "fde"],
    thresholds={"ade": 1.0, "fde": 2.0},
)
```

Some metric functions also have their own parameters. For example, `miss_rate` uses a final-point distance threshold internally. Pass it as a normal keyword argument or through `metric_kwargs`:

```python
# predictions should be a KxTx2 or KxTx3 multimodal trajectory array.
result = evaluator.evaluate(
    prediction=predictions,
    ground_truth=gt,
    metrics=["miss_rate"],
    threshold=0.5,
)
```

## Results

`Evaluator.evaluate()` returns an `EvaluationResult`:

```python
summary = result.summary()
payload = result.to_json()
markdown = result.to_markdown()
frame = result.to_dataframe()
csv_text = result.to_csv()
reloaded = EvaluationResult.from_json(result.to_json())
```

`to_dataframe()` and `to_csv()` require pandas. Install `robometrics[io]` to use
those export paths.

`to_json()` emits standards-compliant JSON. Non-finite metric values such as `NaN` or `inf` are exported as `null`.
Each affected metric includes `metadata["value_serialization"]` so strict JSON
consumers can distinguish `nan`, `inf`, and `-inf` from ordinary null values.

`summary()` includes per-unit summaries when units are mixed. The legacy
aggregate is still present for convenience, but aggregate statistics are `None`
and include a warning when values mix units such as meters and `m^2/s^6`.

Each row is a `MetricResult` with:

- `name`
- `value`
- `unit`
- `passed`
- `threshold`
- `metadata`

If a known metric fails during execution, evaluation continues and the returned `MetricResult` contains `metadata["error"]`. Unknown metric names fail before execution with `UnknownMetricError`, so no partial `EvaluationResult` is returned for misspelled metric names.

Array-valued metrics are reduced to scalar `MetricResult.value` entries when
run through the evaluator. Vector arrays such as acceleration and jerk use mean
row-wise norm. Scalar arrays such as curvature use the mean. The original array
and reduction strategy are stored in metadata.

Use `result.strict_passed` for CI gates that should ignore metrics without
thresholds and fail if any thresholded metric fails. `result.passed` remains
`None` unless every metric defines pass/fail status.

If a metric raises a runtime error (for example, wrong input shape), it is
returned with `value=nan` and `passed=None`. It is counted in
`summary()["error_count"]` and its `metadata["error"]` field describes the
failure. Errors are **explicit and visible** — check `error_count` rather than
assuming silence means success. Because errored metrics have `passed=None`, they
are excluded from `strict_passed` and will not fail a CI gate on their own.

## Dataset Evaluation

`evaluate_dataset()` runs matching prediction and ground-truth sequences through
the same metric selection path, then returns one aggregate `MetricResult` per
metric:

```python
dataset_result = evaluator.evaluate_dataset(
    predictions=[pred_a, pred_b],
    ground_truths=[gt_a, gt_b],
    metrics=["ade", "fde"],
    thresholds={"ade": 0.5, "fde": 1.0},
    bootstrap_ci=1000,
    bootstrap_seed=0,
)
```

Each aggregate result stores the mean in `value` and sample count, finite count,
min, max, standard deviation, raw values, and sample errors in metadata.
When `bootstrap_ci` is set, `bootstrap_seed` controls the random resampling used
for confidence intervals; the default `0` preserves reproducible output, and
passing another integer changes the resampling stream.

## Registry

The default `registry` stores built-in metric metadata and callables:

```python
from robometrics import registry

registry.list_metrics()
registry.get("ade")
registry.get("average_displacement_error")
```

Custom registries can be passed into `Evaluator(metric_registry=...)` for local project-specific metric sets.

For a complete runnable script that separates single-trajectory and multimodal prediction workflows, see `examples/evaluator_usage.py`.

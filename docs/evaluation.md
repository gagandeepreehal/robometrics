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

`prediction` and `ground_truth` are mapped to the argument names used by built-in metrics. Metric-specific inputs can be supplied as keyword arguments, such as `dt`, `actor_trajs`, `ego_radius`, `actor_radius`, `lane_boundary`, `constraints`, `k`, or `threshold`.
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

Pass/fail thresholds are applied to metric result values with `value <= threshold`:

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
```

`to_json()` emits standards-compliant JSON. Non-finite metric values such as `NaN` or `inf` are exported as `null`.
Each affected metric includes `metadata["value_serialization"]` so strict JSON
consumers can distinguish `nan`, `inf`, and `-inf` from ordinary null values.

`summary()` includes per-unit summaries. The legacy aggregate is still present
for convenience, but it includes a warning when it mixes units such as meters
and `m^2/s^6`.

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

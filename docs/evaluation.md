# Evaluation API

RoboMetrics evaluation objects provide a local Python API for running metrics together. They do not require dashboards, services, databases, ROS, simulators, or cloud infrastructure.

## Evaluator

```python
from robometrics import Evaluator

evaluator = Evaluator()
result = evaluator.evaluate(
    prediction=pred,
    ground_truth=gt,
    metrics=["ade", "fde", "miss_rate"],
)
```

`prediction` and `ground_truth` are mapped to the argument names used by built-in metrics. Metric-specific inputs can be supplied as keyword arguments, such as `dt`, `actor_trajs`, `ego_radius`, `actor_radius`, `lane_boundary`, `constraints`, `k`, or `threshold`.

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

Each row is a `MetricResult` with:

- `name`
- `value`
- `unit`
- `passed`
- `threshold`
- `metadata`

If a metric fails during execution, evaluation continues and the returned `MetricResult` contains `metadata["error"]`.

## Registry

The default `registry` stores built-in metric metadata and callables:

```python
from robometrics import registry

registry.list_metrics()
registry.get("ade")
registry.get("average_displacement_error")
```

Custom registries can be passed into `Evaluator(metric_registry=...)` for benchmark profiles or local project-specific metric sets.

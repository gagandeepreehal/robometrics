# RoboMetrics

[![CI](https://github.com/robometrics/robometrics/actions/workflows/ci.yml/badge.svg)](https://github.com/robometrics/robometrics/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![PyPI](https://img.shields.io/badge/PyPI-not%20released-lightgrey)

Evaluation metrics for Physical AI.

RoboMetrics is a lightweight Python package for evaluating trajectories, predictions, comfort, safety, and physical consistency in robotics and autonomy workflows. It is local-first: no simulator, ROS, GPU, cloud service, database, or dashboard is required.

## Why This Exists

Physical AI projects often start with scattered metric functions and ad hoc evaluation scripts. RoboMetrics provides a small common layer for:

- reusable metric functions,
- a registry of named metrics,
- an `Evaluator` for running multiple metrics together,
- structured results with thresholds,
- JSON, Markdown, and pandas DataFrame exports.

RoboMetrics is a public alpha candidate. It is not production-proven or an industry standard.

## Installation

```bash
pip install robometrics
```

For local development:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check .
mypy robometrics
```

## Quickstart

```python
import numpy as np

from robometrics import Evaluator, average_displacement_error

pred = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
gt = np.array([[0.0, 0.0], [1.1, 0.0], [2.2, 0.0]])

ade = average_displacement_error(pred, gt)

result = Evaluator().evaluate(
    prediction=pred,
    ground_truth=gt,
    metrics=["ade", "fde"],
    thresholds={"ade": 0.5, "fde": 1.0},
)

print("ADE:", ade)
print(result.summary())
print(result.to_markdown())
```

Runnable examples:

```bash
python examples/basic_metrics.py
python examples/evaluator_quickstart.py
python examples/thresholds.py
python examples/export_results.py
```

## Core Concepts

- **Metric functions** are plain Python functions such as `average_displacement_error(pred, gt)`.
- **Registry** maps stable names such as `ade` and `fde` to metric functions and metadata.
- **Evaluator** runs named metrics or categories against supplied inputs.
- **MetricResult** stores one metric value, unit, threshold, pass/fail status, and metadata.
- **EvaluationResult** stores a collection of metric results and supports summaries and exports.

## Supported Metric Categories

- **Trajectory:** ADE, FDE, Hausdorff distance, path length, curvature, lateral error, longitudinal error.
- **Prediction:** minADE, minFDE, miss rate, top-k trajectory error.
- **Comfort/control:** acceleration, jerk, jerk cost, max acceleration, max deceleration, smoothness score.
- **Safety:** collision rate, time to collision, minimum distance to actors, lane departure rate.
- **Physical consistency:** speed profile, acceleration/jerk/curvature limit checks, dynamic feasibility score.

Most metrics return a `float`. Thresholded physical consistency checks return `MetricResult`.

## Evaluator Example

```python
from robometrics import Evaluator

result = Evaluator().evaluate(
    prediction=pred,
    ground_truth=gt,
    metrics=["ade", "fde"],
)
```

Use `metrics="all"` to run every compatible registered metric, or select groups:

```python
result = Evaluator().evaluate(
    prediction=pred,
    ground_truth=gt,
    categories=["trajectory"],
)
```

Known metric execution failures are returned as failed metric results with `metadata["error"]`. Unknown metric names raise `UnknownMetricError` before evaluation starts.

## Registry Example

```python
from robometrics import registry

print(registry.list_metrics())
print(registry.get("ade"))
print(registry.get("average_displacement_error"))
```

Registered aliases include:

- `average_displacement_error` -> `ade`
- `final_displacement_error` -> `fde`
- `minade` -> `min_ade`
- `minfde` -> `min_fde`

## Threshold Example

```python
result = Evaluator().evaluate(
    prediction=pred,
    ground_truth=gt,
    metrics=["ade", "fde"],
    thresholds={"ade": 0.5, "fde": 1.0},
)

for metric in result.results:
    print(metric.name, metric.value, metric.threshold, metric.passed)
```

## Export Example

```python
payload = result.to_dict()
json_text = result.to_json()
markdown = result.to_markdown()
frame = result.to_dataframe()
```

`to_json()` emits standards-compliant JSON. Non-finite metric values such as `NaN` and `inf` are exported as `null`.

## Input Format

Trajectory metrics accept finite, non-empty `Nx2` or `Nx3` NumPy-compatible arrays. Prediction metrics accept `KxTx2` or `KxTx3` predictions and `Tx2` or `Tx3` ground truth. Invalid shapes, empty inputs, mismatched lengths, and non-finite values raise `ValueError`.

Single-point trajectories are accepted by metrics with well-defined degenerate outputs. `Nx3` trajectories are supported wherever `Nx2` trajectories are accepted. `min_distance_to_actors(ego_traj, [])` returns `math.inf`.

File read and parse failures raise `TrajectoryIOError`. Shape, missing-column, and invalid-value validation issues raise `ValueError`.

## Roadmap

- Expand metric coverage with focused, well-tested additions.
- Add benchmark profile helpers around the existing evaluator objects.
- Improve documentation examples and API reference coverage.
- Keep the package local-first and dependency-light.

Not planned for this initial release: dashboards, web apps, databases, cloud services, simulators, ROS integrations, or leaderboard infrastructure.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/contributing.md](docs/contributing.md).

Before opening a pull request:

```bash
ruff check .
mypy robometrics
pytest --cov=robometrics --cov-report=term-missing
python examples/basic_metrics.py
python examples/evaluator_quickstart.py
python examples/thresholds.py
python examples/export_results.py
```

## License

MIT. See [LICENSE](LICENSE).

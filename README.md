# RoboMetrics

[![CI](https://github.com/robometrics/robometrics/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/robometrics/robometrics/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![PyPI](https://img.shields.io/badge/PyPI-not%20released-lightgrey)

Lightweight robotics metrics for Python.

RoboMetrics is a small local Python library for computing robotics trajectory,
prediction, safety, comfort, and physics metrics from NumPy arrays and simple
CSV/JSON trajectory files.

## Why This Exists

Robotics projects often grow scattered metric functions across notebooks,
experiments, and scripts. RoboMetrics keeps common metrics in one typed,
dependency-light package that can be imported directly inside local robotics
codebases.

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

from robometrics import ade, fde, path_length

prediction = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
ground_truth = np.array([[0.0, 0.0], [1.1, 0.0], [2.2, 0.0]])

print("ADE (m):", ade(prediction, ground_truth))
print("FDE (m):", fde(prediction, ground_truth))
print("Path length (m):", path_length(ground_truth))
```

Runnable examples:

```bash
python examples/trajectory_metrics.py
python examples/prediction_metrics.py
python examples/safety_metrics.py
python examples/comfort_metrics.py
python examples/load_from_csv.py
```

## Input Shapes

Trajectory-like inputs are NumPy-compatible arrays with finite numeric values:

- Single trajectory: `Nx2` or `Nx3`
- Multimodal predictions: `KxTx2` or `KxTx3`
- Ground-truth trajectory for prediction metrics: `Tx2` or `Tx3`
- Actor trajectories for safety metrics: a list of `Nx2` or `Nx3` arrays

`N` or `T` is the number of timesteps, `K` is the number of prediction modes,
and columns are position coordinates in meters. `Nx3` inputs are supported by
metrics that operate on positions; distance-based metrics use all available
coordinate dimensions unless documented otherwise.

Empty arrays, NaN/inf values, bad ranks, mismatched trajectory lengths, invalid
`dt`, and incompatible dimensions raise `ValueError` with a targeted message.

## Units

- Position and distance inputs: meters
- Time step `dt`: seconds
- Speed: meters per second
- Acceleration: meters per second squared
- Jerk: meters per second cubed
- Curvature: inverse meters
- Rates and scores: unitless floats

## Return Types

Most public metric functions return `float` or `numpy.ndarray`. Threshold-style
physics helpers return `MetricResult`, which stores a value, unit, optional
threshold, pass/fail status, and metadata.

`EvaluationResult` is a small container for local batches of metric results and
can export dictionaries, strict JSON, Markdown tables, and pandas DataFrames.
Non-finite metric values are serialized as `null` in JSON.

## Metric Examples

### Trajectory

```python
import numpy as np

from robometrics import curvature, path_length

trajectory = np.array([[0.0, 0.0], [1.0, 0.2], [2.0, 0.5]])

print(path_length(trajectory))  # meters
print(curvature(trajectory))    # 1/meters
```

### Prediction

```python
import numpy as np

from robometrics import min_ade, min_fde, miss_rate

predictions = np.array(
    [
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        [[0.0, 0.0], [1.1, 0.0], [2.1, 0.0]],
    ]
)
ground_truth = np.array([[0.0, 0.0], [1.0, 0.1], [2.0, 0.2]])

print(min_ade(predictions, ground_truth))
print(min_fde(predictions, ground_truth))
print(miss_rate(predictions, ground_truth, threshold=0.5))
```

### Safety

```python
import numpy as np

from robometrics import collision_rate, min_distance_to_actors

ego = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
actors = [np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.4]])]

print(min_distance_to_actors(ego, actors))                 # meters
print(collision_rate(ego, actors, ego_radius=0.3, actor_radius=0.3))
```

### Comfort

```python
import numpy as np

from robometrics import acceleration, jerk, jerk_cost

trajectory = np.array([[0.0, 0.0], [1.0, 0.1], [2.0, 0.4], [3.0, 0.9]])
dt = 0.5

print(acceleration(trajectory, dt=dt))  # m/s^2
print(jerk(trajectory, dt=dt))          # m/s^3
print(jerk_cost(trajectory, dt=dt))
```

### Physics

```python
import numpy as np

from robometrics import acceleration_limits_violated, dynamic_feasibility_score

trajectory = np.array([[0.0, 0.0], [1.0, 0.1], [2.0, 0.3], [3.0, 0.6]])

print(acceleration_limits_violated(trajectory, dt=0.5, max_accel=5.0))
print(dynamic_feasibility_score(trajectory, dt=0.5))
```

## CSV And JSON

```python
from robometrics.io import load_trajectory_csv, load_trajectory_json

csv_traj = load_trajectory_csv("trajectory.csv")
json_traj = load_trajectory_json("trajectory.json")
```

CSV files must include `x` and `y` columns, and may include `z`. JSON files may
contain either a raw list of points or an object with a `points` field.

## Lightweight Evaluator And Registry

The evaluator and registry are intentionally small helpers for local scripts.
Use them when named metric selection or threshold reporting is useful:

```python
from robometrics import Evaluator, registry

print(registry.list_metrics())

result = Evaluator().evaluate(
    prediction=prediction,
    ground_truth=ground_truth,
    metrics=["ade", "fde"],
    thresholds={"ade": 0.5, "fde": 1.0},
)

print(result.to_json())
```

Unknown metric names raise `UnknownMetricError` before evaluation starts.
Metric execution failures are returned as failed `MetricResult` entries with
`metadata["error"]`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/contributing.md](docs/contributing.md).

Before opening a pull request:

```bash
ruff check .
mypy robometrics
pytest --cov=robometrics --cov-report=term-missing
python examples/trajectory_metrics.py
python examples/prediction_metrics.py
python examples/safety_metrics.py
python examples/comfort_metrics.py
python examples/load_from_csv.py
```

## License

MIT. See [LICENSE](LICENSE).

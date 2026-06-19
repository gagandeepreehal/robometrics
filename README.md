# RoboMetrics

[![CI](https://github.com/gagandeepreehal/robometrics/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/gagandeepreehal/robometrics/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![PyPI](https://img.shields.io/pypi/v/robometrics.svg)](https://pypi.org/project/robometrics/)

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
pip install "robometrics[io]"  # CSV loading and pandas exports
```

RoboMetrics is primarily a Python library. The installed CLI is intentionally
small and meant for package verification and metric discovery:

```bash
python -m robometrics --help
robometrics list-metrics
robometrics list-metrics --format json
robometrics describe ade
robometrics version
```

If a user-level pip install places `robometrics` outside `PATH`, use
`python -m robometrics ...` or add the script directory reported by pip, such as
`$HOME/Library/Python/3.9/bin` on macOS system Python, to `PATH`.

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
metrics that operate on positions. ADE, FDE, Hausdorff distance, prediction
distance metrics, and path length use all coordinate dimensions. Curvature,
lateral error, longitudinal error, collision checks, lane departure, and
constant-velocity TTC are planar XY metrics.

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

Most public metric functions return `float`; profile-style functions such as
`curvature()`, `speed_profile()`, `acceleration()`, and `jerk()` return per-step
NumPy arrays. For acceleration or jerk magnitudes, use
`acceleration_magnitude()` and `jerk_magnitude()`; scalar helpers such as
`mean_curvature()`, `mean_acceleration()`, and `rms_acceleration()` are provided
for common CI summaries. Threshold-style physics helpers return
`MetricResult`, which stores a value, unit, optional threshold, pass/fail status,
and metadata.

`EvaluationResult` is a small container for local batches of metric results and
can export and reload dictionaries, strict JSON, Markdown tables, CSV, and
pandas DataFrames. Install `robometrics[io]` for CSV and pandas-backed exports.
Non-finite metric values are serialized as `null` in JSON with metadata that
records whether the original value was `nan`, `inf`, or `-inf`.

## Metric Categories

The registry keeps a pragmatic taxonomy:

- `trajectory`: geometric path and path-comparison metrics.
- `prediction`: multimodal forecast metrics such as minADE, minFDE, miss indicator, and top-k error.
- `comfort`: kinematic profiles and smoothness values commonly used for ride or control quality.
- `safety`: collision, distance-to-actor, lane, and TTC metrics.
- `physics`: thresholded dynamic feasibility checks and limit-result helpers.

Some physical quantities appear in more than one category by design. For
example, `acceleration()` returns a comfort/control profile, while
`acceleration_limits_violated()` is a physics limit check.

## Metric Examples

### Trajectory

```python
import numpy as np

from robometrics import curvature, path_length

trajectory = np.array([[0.0, 0.0], [1.0, 0.2], [2.0, 0.5]])

print(path_length(trajectory))  # meters
print(curvature(trajectory))    # per-step planar XY curvature, 1/meters
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

Prediction modes passed to `topk_trajectory_error()` should already be ranked
by confidence, with highest confidence first. `miss_rate()` returns a per-sample
0.0/1.0 miss indicator; average it across a dataset for a conventional miss
rate.

### Safety

```python
import numpy as np

from robometrics import collision_rate, min_distance_to_actors

ego = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
actors = [np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.4]])]

print(min_distance_to_actors(ego, actors))                 # meters
print(collision_rate(ego, actors, ego_radius=0.3, actor_radius=0.3))
```

`collision_rate()` divides by ego timesteps covered by at least one actor
trajectory. If an actor has only two timesteps, only those two aligned ego
timesteps contribute to the denominator.

### Comfort

```python
import numpy as np

from robometrics import acceleration, jerk, jerk_cost, smoothness_score

trajectory = np.array([[0.0, 0.0], [1.0, 0.1], [2.0, 0.4], [3.0, 0.9]])
dt = 0.5

print(acceleration(trajectory, dt=dt))  # m/s^2
print(jerk(trajectory, dt=dt))          # m/s^3
print(jerk_cost(trajectory, dt=dt))
print(smoothness_score(trajectory))
```

`smoothness_score()` uses `1 / (1 + log1p(cost))`, where `cost` is the mean
squared third finite difference normalized by mean squared step length. The
score is unitless, invariant to coordinate scale, and separate from physical
`jerk_cost(traj, dt)`. It returns `1.0` for trajectories with fewer than four
points because third finite differences are not measurable.

### Physics

```python
import numpy as np

from robometrics import acceleration_limits_violated, dynamic_feasibility_score

trajectory = np.array([[0.0, 0.0], [1.0, 0.1], [2.0, 0.3], [3.0, 0.6]])

print(acceleration_limits_violated(trajectory, dt=0.5, max_accel=5.0))
print(dynamic_feasibility_score(trajectory, dt=0.5, constraints={"max_accel": 5.0}))
```

`dynamic_feasibility_score()` uses the worst relative violation across supplied
limits, so adding satisfied constraints does not dilute an existing violation.
Zero limits are accepted; positive observed motion against a zero limit scores
`0.0`.

## CSV And JSON

```python
from robometrics.io import load_trajectory_csv, load_trajectory_json

csv_traj = load_trajectory_csv("trajectory.csv")
json_traj = load_trajectory_json("trajectory.json")
```

CSV files must include `x` and `y` columns, and may include `z`. JSON files may
contain either a raw list of points or an object with a `points` field.
Use `load_trajectory_dir()` to load every supported trajectory file in a
directory into a filename-keyed dictionary.

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
reloaded = result.from_json(result.to_json())
```

Unknown metric names raise `UnknownMetricError` before evaluation starts.
Metric execution failures are returned as failed `MetricResult` entries with
`metadata["error"]`.

When array-valued metrics are run through the evaluator, vectors are reduced to
mean row-wise norm and scalar arrays are reduced to their mean. The raw value
and reduction name are stored in metric metadata. `EvaluationResult.summary()`
includes per-unit summaries when units mix; in that case the legacy aggregate
includes a warning and leaves aggregate statistics as `None`. Use
`result.strict_passed` for CI gates that should ignore metrics without
thresholds while still failing on any thresholded metric failure.

For dataset-level aggregation, pass matching prediction and ground-truth
sequences to `Evaluator.evaluate_dataset(...)`. It returns one aggregate
`MetricResult` per metric with per-sample values and count/min/max/std metadata.

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

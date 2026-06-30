# RoboMetrics

[![CI](https://github.com/gagandeepreehal/robometrics/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/gagandeepreehal/robometrics/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Coverage](https://img.shields.io/badge/coverage-90%25%20minimum-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
[![PyPI](https://img.shields.io/pypi/v/robometrics.svg)](https://pypi.org/project/robometrics/)
[![Downloads](https://static.pepy.tech/badge/robometrics)](https://pepy.tech/project/robometrics)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://gagandeepreehal.github.io/robometrics/)

Lightweight robotics metrics for Python.

[Documentation](https://gagandeepreehal.github.io/robometrics/) |
[Colab demo](https://colab.research.google.com/github/gagandeepreehal/robometrics/blob/main/notebooks/robometrics_colab_demo.ipynb)

RoboMetrics is a small local Python library for computing robotics trajectory,
prediction, temporal drift, safety, comfort, coverage, calibration, physics,
and diversity metrics from NumPy arrays and simple CSV/JSON trajectory files.

It is a trusted metrics layer, not a simulator, dashboard platform, dataset
host, robotics framework, or leaderboard service. Use it when you need typed,
repeatable metric functions, CI regression checks, lightweight trajectory file
evaluation, or project-local metric packs. Do not use it as a replacement for
scenario generation, physics simulation, dataset storage, or full benchmark
governance.

## Why This Exists

Robotics projects often grow scattered metric functions across notebooks,
experiments, and scripts. RoboMetrics keeps common metrics in one typed,
dependency-light package that can be imported directly inside local robotics
codebases.

## Installation

```bash
pip install robometrics
pip install "robometrics[io]"  # CSV loading and pandas exports
pip install "robometrics[mcap]"  # optional MCAP JSON-message adapter
pip install "robometrics[loggers]"  # optional W&B and MLflow logging
```

RoboMetrics is primarily a Python library. The installed CLI is intentionally
local-first and file-based. The examples below use `python -m robometrics`
because it works even when a user-level pip install places console scripts
outside `PATH`.

The `robometrics` console script is also installed; use it directly when the
script directory reported by pip is on `PATH`.

```bash
python -m robometrics --help
python -m robometrics list-metrics
python -m robometrics list-metrics --format json
python -m robometrics describe ade

cat > predictions.csv <<'CSV'
t,x,y
0,0.0,0.0
1,1.0,0.0
2,2.0,0.0
CSV

cat > ground_truth.csv <<'CSV'
t,x,y
0,0.0,0.0
1,1.1,0.0
2,2.1,0.0
CSV

python -m robometrics validate predictions.csv
python -m robometrics evaluate \
  --pred predictions.csv \
  --gt ground_truth.csv \
  --metrics ade fde \
  --threshold ade=0.5 \
  --output result.json
python -m robometrics report result.json --output report.html
python -m robometrics benchmark list
python -m robometrics version
```

For local development:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest
ruff check .
mypy robometrics
```

## Quickstart

Five-minute local workflow:

```bash
pip install "robometrics[io]"

cat > predictions.csv <<'CSV'
t,x,y
0,0.0,0.0
1,1.0,0.0
2,2.0,0.0
CSV

cat > ground_truth.csv <<'CSV'
t,x,y
0,0.0,0.0
1,1.1,0.0
2,2.1,0.0
CSV

python -m robometrics validate predictions.csv
python -m robometrics evaluate \
  --pred predictions.csv \
  --gt ground_truth.csv \
  --metrics ade fde \
  --threshold ade=0.5 \
  --threshold fde=1.0 \
  --output result.json
python -m robometrics report result.json --output report.html
```

Python API:

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
python examples/basic_metrics.py
python examples/evaluator_quickstart.py
python examples/thresholds.py
python examples/export_results.py
python examples/trajectory_metrics.py
python examples/prediction_metrics.py
python examples/driving_metrics.py
python examples/safety_metrics.py
python examples/comfort_metrics.py
python examples/new_metrics_example.py
python examples/load_from_csv.py
python examples/dataset_loader_evaluation.py
python examples/evaluator_usage.py
python examples/manipulation_metrics.py
```

## Input Shapes

Trajectory-like inputs are NumPy-compatible arrays with finite numeric values:

- Single trajectory: `Nx2` or `Nx3`
- Multimodal predictions: `KxTx2` or `KxTx3`
- Ground-truth trajectory for prediction metrics: `Tx2` or `Tx3`
- Actor trajectories for safety metrics: a list of `Nx2` or `Nx3` arrays
- General time-series metrics: `TxD` or `BxTxD`
- Coverage samples and behavior embeddings: `NxD`
- Batched behavior trajectories or actions: `NxTxD`

`N` or `T` is the number of timesteps, `K` is the number of prediction modes,
and columns are position coordinates in meters. `Nx3` inputs are supported by
metrics that operate on positions. ADE, FDE, Hausdorff distance, prediction
distance metrics, and path length use all coordinate dimensions. Curvature,
lateral error, longitudinal error, collision checks, lane departure, and
constant-velocity TTC are planar XY metrics.

Empty arrays, NaN/inf values, bad ranks, invalid `dt`, and incompatible
dimensions raise `ValueError` with a targeted message. Metrics that compare
aligned samples, such as ADE and FDE, require matching trajectory shapes.
Set-based metrics such as `hausdorff_distance` may compare different numbers
of points when coordinate dimensionality matches.
The `Evaluator` also accepts `Trajectory` schema objects and converts them with
`.array()` before dispatching metric functions.

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
Evaluation JSON includes top-level `"schema_version": "1"`; downstream CI,
dashboards, and experiment trackers should treat that as the stable output
contract and reject unknown future schema versions instead of guessing.
Non-finite metric values are serialized as `null` in JSON with metadata that
records whether the original value was `nan`, `inf`, or `-inf`.
Use `result.log_to_wandb(run)` or `result.log_to_mlflow(run)` to send finite
metric values and summary counts into existing experiment runs. Passing a run
or logger object avoids importing optional logging packages; install
`robometrics[wandb]`, `robometrics[mlflow]`, or `robometrics[loggers]` when you
want RoboMetrics to import those tools directly.

`speed_profile()` returns one speed estimate per input point; endpoint speeds
are finite-difference gradient estimates, not `N-1` interval speeds.
`curvature()` assumes uniformly spaced samples; resample irregular timestamped
paths before using curvature or curvature-derived metrics.

## Metric Categories

The registry keeps a pragmatic taxonomy:

- `trajectory`: geometric path and path-comparison metrics.
- `prediction`: multimodal forecast metrics such as minADE, minFDE, miss indicator, and top-k error.
- `temporal`: rollout drift, control-sequence smoothness, and compounding-error metrics.
- `comfort`: kinematic profiles and smoothness values commonly used for ride or control quality.
- `safety`: collision, distance-to-actor, lane, and TTC metrics.
- `coverage`: grid coverage over state or action samples.
- `calibration`: expected calibration error for confidence predictions.
- `physics`: thresholded dynamic feasibility checks and limit-result helpers.
- `diversity`: behavior diversity over embeddings, trajectories, or action sequences.

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

from robometrics import (
    AgentState,
    collision_rate,
    collision_rate_obb,
    min_distance_to_actors,
    time_to_collision,
)

ego = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
actors = [np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.4]])]

print(min_distance_to_actors(ego, actors))                 # meters
print(collision_rate(ego, actors, ego_radius=0.3, actor_radius=0.3))

ego_state = AgentState(x=0.0, y=0.0, vx=2.0, vy=0.0, radius=0.3)
actor_state = AgentState(x=10.0, y=0.0, vx=0.0, vy=0.0, radius=0.3)
print(time_to_collision(ego_state, actor_state))           # seconds

ego_dims = np.array([4.5, 2.0])                            # length, width
ego_yaws = np.array([0.0, 0.0, 0.0])
actor_dims = [np.array([4.5, 2.0])]
actor_yaws = [np.array([0.0, 0.0, 0.0])]
print(collision_rate_obb(ego, ego_dims, ego_yaws, actors, actor_dims, actor_yaws))
```

`collision_rate()` divides by ego timesteps covered by at least one actor
trajectory. If an actor has only two timesteps, only those two aligned ego
timesteps contribute to the denominator.
`time_to_collision()` accepts `AgentState`, dict, flat `[x, y, vx, vy, radius]`
state arrays, or trajectory arrays when `dt` is supplied. For trajectory inputs,
the first segment estimates each agent's constant velocity.
`collision_rate_obb()` uses oriented bounding boxes with dimensions ordered as
`[length, width]` and yaw angles in radians.

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
points with a `RuntimeWarning` because third finite differences are not
measurable. `acceleration()` and `jerk()` suppress finite-difference roundoff
noise below `1e-10`.

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
from robometrics import load_trajectory_csv, load_trajectory_json

csv_traj = load_trajectory_csv("trajectory.csv")
json_traj = load_trajectory_json("trajectory.json")
```

CSV files must include `x` and `y` columns, and may include `z`. The generic
CSV adapter preserves a present `z` column when constructing `Trajectory`
objects. JSON files may contain either a raw list of points or an object with a
`points` field.
Use `load_trajectory_dir()` to load every supported trajectory file in a
directory into a filename-keyed dictionary.

## Lightweight Evaluator And Registry

The evaluator and registry are intentionally small helpers for local scripts.
Use them when named metric selection or threshold reporting is useful:

```python
from robometrics import EvaluationResult, Evaluator, registry

print(registry.list_metrics())

result = Evaluator().evaluate(
    prediction=prediction,
    ground_truth=ground_truth,
    metrics=["ade", "fde"],
    thresholds={"ade": 0.5, "fde": 1.0},
)

print(result.to_json())
reloaded = EvaluationResult.from_json(result.to_json())
```

PyTorch-style tensor inputs are accepted without adding PyTorch as a dependency:
objects with `.detach().cpu().numpy()` are converted before metric dispatch.

Unknown metric names raise `UnknownMetricError` before evaluation starts.
Metric execution failures are returned as `MetricResult` entries with
`value=nan`, `passed=None`, and `metadata["error"]`. They are visible through
`EvaluationResult.summary()["error_count"]` but are ignored by
`result.strict_passed` unless a thresholded metric actually fails.

When array-valued metrics are run through the evaluator, vectors are reduced to
mean row-wise norm and scalar arrays are reduced to their mean. The raw value
and reduction name are stored in metric metadata. `EvaluationResult.summary()`
includes per-unit summaries when units mix; in that case the legacy aggregate
includes a warning and leaves aggregate statistics as `None`. Use
`result.strict_passed` for CI gates that should ignore metrics without
thresholds while still failing on any thresholded metric failure.
Thresholds follow registry directionality: lower-is-better metrics pass with
`value <= threshold`, while metrics marked `higher_is_better=True` pass with
`value >= threshold`.

For dataset-level aggregation, pass matching prediction and ground-truth
sequences to `Evaluator.evaluate_dataset(...)`. It returns one aggregate
`MetricResult` per metric with per-sample values and count/min/max/std metadata.

## CLI Evaluation, Validation, Reports, And Profiles

`python -m robometrics evaluate` loads CSV/JSON trajectories, validates file
existence, runs named metrics using each metric's registry compatibility rules,
and writes strict `EvaluationResult` JSON with metric values, metadata,
timestamp, thresholds, and pass/fail status. Metrics such as ADE and FDE require
matching shapes; `hausdorff_distance` can compare unequal sample counts when
both trajectories use the same coordinate dimensionality.

```bash
cat > predictions.csv <<'CSV'
t,x,y
0,0.0,0.0
1,1.0,0.0
2,2.0,0.0
CSV

cat > ground_truth.csv <<'CSV'
t,x,y
0,0.0,0.0
1,1.1,0.0
2,2.1,0.0
CSV

python -m robometrics evaluate \
  --pred predictions.csv \
  --gt ground_truth.csv \
  --metrics ade fde \
  --threshold ade=0.5 \
  --output result.json
```

`python -m robometrics validate <path>` inspects trajectory-style CSV/JSON files for
missing fields, invalid numeric values, NaN/inf, inconsistent dimensions,
non-monotonic timestamps, empty trajectories, and unsupported formats. Add
`--output validation.json` for machine-readable output.

`python -m robometrics report result.json --output report.html` generates a lightweight
static HTML report with summary, metric values, pass/fail indicators, metadata,
and baseline comparison metadata when present.

Benchmark profiles package repeatable metric sets for small local gates:

```bash
python -m robometrics benchmark list
python -m robometrics benchmark run policy_regression_ci \
  --pred predictions.csv \
  --gt ground_truth.csv \
  --output result.json
```

Built-in profiles are `trajectory_prediction_basic`, `mobile_robot_safety`,
`manipulation_tracking`, and `policy_regression_ci`. They are smoke/regression
profiles, not leaderboard definitions.

## Dataset Adapters

The `robometrics.adapters` package provides lightweight adapters with a common
interface: `load(path)`, `validate(path)`, and `metadata(path)`. Built-ins cover
generic CSV, generic JSON, ROS-style JSON exports without importing ROS,
ROS 2 bag JSON exports without importing `rclpy`, LeRobot-style JSON exports
without importing LeRobot, RLDS-style JSON exports without TensorFlow, and MCAP
JSON-message logs through the optional `robometrics[mcap]` extra. The generic
CSV adapter loads `x`/`y` and preserves `z` when that optional column is
present. Heavy robotics or dataset packages are not hard dependencies.

## Metric Packs

External packages can expose a `METRIC_PACK` list and register it with
`robometrics.load_pack("module_name")`. Each entry uses the same registry
contract as built-in metrics: name, callable, category, unit, required inputs,
default kwargs, aliases, reference, and directionality. See
`examples/custom_metric_pack.py` and the metric-pack guide for a complete
example.

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
python examples/trajectory_metrics.py
python examples/prediction_metrics.py
python examples/driving_metrics.py
python examples/safety_metrics.py
python examples/comfort_metrics.py
python examples/new_metrics_example.py
python examples/load_from_csv.py
python examples/dataset_loader_evaluation.py
python examples/evaluator_usage.py
python examples/manipulation_metrics.py
```

## License

MIT. See [LICENSE](LICENSE).

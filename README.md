# RoboMetrics

RoboMetrics is a lightweight, standalone Python metrics library for Physical AI systems. It is designed to be imported inside robotics, autonomy, drones, and manipulation codebases without requiring simulators, robots, large models, GPUs, ROS, cloud services, or a dashboard.

The first version focuses on reusable NumPy-based metrics for trajectory evaluation, prediction evaluation, planning/control smoothness, safety checks, and physical consistency checks.

## Installation

```bash
pip install robometrics
```

For local development:

```bash
pip install -e ".[dev]"
pytest
ruff check .
mypy robometrics
```

## Quickstart

```python
import numpy as np

from robometrics import Evaluator, average_displacement_error, collision_rate, jerk_cost

pred = np.array([[0, 0], [1, 0], [2, 0]])
gt = np.array([[0, 0], [1.1, 0], [2.1, 0]])

ade = average_displacement_error(pred, gt)
comfort = jerk_cost(pred, dt=0.1)
collisions = collision_rate(
    ego_traj=pred,
    actor_trajs=[np.array([[10, 0], [10, 0], [10, 0]])],
    ego_radius=0.5,
    actor_radius=0.5,
)

evaluator = Evaluator()
result = evaluator.evaluate(
    prediction=pred,
    ground_truth=gt,
    metrics=["ade", "fde"],
    thresholds={"ade": 1.0, "fde": 2.0},
)
print(result.summary())
print(result.to_markdown())
```

## Metric Categories

- Trajectory: ADE, FDE, Hausdorff distance, path length, curvature, lateral error, longitudinal error.
- Prediction: minADE, minFDE, miss rate, top-k trajectory error.
- Comfort/control: acceleration, jerk, jerk cost, max acceleration, max deceleration, smoothness score.
- Safety: collision rate, time to collision, minimum distance to actors, lane departure rate.
- Physical consistency: speed profile, acceleration/jerk/curvature limit checks, dynamic feasibility score.

Most metrics return a `float`. Thresholded physical consistency checks return `MetricResult` with `name`, `value`, `unit`, `passed`, `threshold`, and `metadata`.

## Evaluation API

RoboMetrics also includes local evaluation objects for running multiple metrics together:

- `registry`: central registry for built-in metric metadata and callables.
- `Evaluator`: selects, validates, and runs metrics by name or category.
- `MetricResult`: one metric value plus unit, threshold/pass status, and metadata.
- `EvaluationResult`: collection of results with `summary()`, `to_json()`, `to_markdown()`, and `to_dataframe()`.

Use `metrics="all"` to run every compatible registered metric for the provided inputs, or use `categories=["trajectory"]` to select a group.

## Input Format

Trajectory metrics accept finite, non-empty `Nx2` or `Nx3` NumPy-compatible arrays. Prediction metrics accept `KxTx2` or `KxTx3` predictions and `Tx2` or `Tx3` ground truth. Invalid shapes, empty inputs, mismatched lengths, and NaN values raise `ValueError`.

Simple IO helpers support:

- in-memory NumPy arrays
- `.npy` and `.npz`
- CSV files with `x` and `y` columns
- JSON files with the following shape:

```json
{
  "trajectory": [
    {"t": 0.0, "x": 0.0, "y": 0.0},
    {"t": 0.1, "x": 0.2, "y": 0.0}
  ]
}
```

## Contributing

RoboMetrics should stay small, typed, simulator-independent, and dependency-light. New metrics should include:

- a clear function-level API
- validation for shape and NaN edge cases
- unit tests for normal and edge cases
- documentation updates

Run the local quality checks before opening a pull request:

```bash
pytest
ruff check .
mypy robometrics
```

## License

MIT

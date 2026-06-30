# Quickstart

Use this page when you already have NumPy-like arrays or small CSV/JSON files
and want a first local evaluation. For simulator, ROS, LeRobot, or policy-log
exports, start with [Run From Policy Output](guides/policy_output.md).

## Install

```bash
pip install robometrics
```

Install optional CSV/DataFrame helpers when you need pandas-backed exports:

```bash
pip install "robometrics[io]"
```

## Evaluate One Trajectory

```python
import numpy as np

from robometrics import Evaluator, ade, collision_rate, jerk_cost

pred = np.array([[0, 0], [1, 0], [2, 0]])
gt = np.array([[0, 0], [1.1, 0], [2.1, 0]])

distance_error = ade(pred, gt)
comfort = jerk_cost(pred, dt=0.1)
collisions = collision_rate(pred, [], ego_radius=0.5, actor_radius=0.5)

evaluator = Evaluator()
result = evaluator.evaluate(
    prediction=pred,
    ground_truth=gt,
    metrics=["ade", "fde"],
    thresholds={"ade": 1.0, "fde": 2.0},
)

print(result.summary())
print(result.to_markdown())
print(distance_error, comfort, collisions)
```

## Run From The Shell

```bash
python -m robometrics list-metrics
python -m robometrics describe ade
python -m robometrics evaluate \
  --pred examples/fixtures/predictions.csv \
  --gt examples/fixtures/ground_truth.csv \
  --metrics ade fde \
  --threshold ade=0.5 \
  --threshold fde=1.0 \
  --output result.json
```

The `robometrics` console script is also installed, but `python -m robometrics`
is more reliable in user-level installs where console scripts may be outside
`PATH`.

## Input Shapes

| Input type | Accepted shape | Notes |
| --- | --- | --- |
| Trajectory | `Nx2` or `Nx3` | Finite, non-empty arrays. `Nx3` works wherever trajectory metrics accept `Nx2`. |
| Multimodal prediction | `KxTx2` or `KxTx3` | Used by prediction metrics such as `min_ade`, `min_fde`, and `miss_rate`. |
| Time series | `TxD` or `BxTxD` | Used by temporal and action/control metrics. |
| Samples | `NxD` | Used by coverage and diversity metrics. |

Single-point trajectories are accepted by metrics with well-defined degenerate
outputs. File I/O helpers raise `TrajectoryIOError` for missing or malformed
files, and `ValueError` for invalid trajectory contents.

## Select Metrics

Use `metrics="all"` to run every compatible registered metric, or select groups
with categories such as `["trajectory"]`, `["temporal"]`, `["safety"]`, or
`["physics"]`.

```python
result = evaluator.evaluate(
    prediction=pred,
    ground_truth=gt,
    categories=["trajectory"],
)
```

## Next Steps

<div class="rm-grid rm-grid-2" markdown="1">
<div class="rm-card" markdown="1">
[Metric Catalog](metrics.md)

Browse canonical metric names, aliases, units, directionality, and edge cases.
</div>

<div class="rm-card" markdown="1">
[Evaluation Guide](evaluation.md)

Learn thresholds, `strict_passed`, dataset aggregation, JSON output, and registry behavior.
</div>

<div class="rm-card" markdown="1">
[CLI Commands](cli.md)

Validate files, evaluate CSV/JSON inputs, generate reports, and run benchmark profiles.
</div>

<div class="rm-card" markdown="1">
[What RoboMetrics Is And Isn't](what_it_is.md)

Check the boundary between a metrics library, simulator, benchmark, and dashboard.
</div>
</div>

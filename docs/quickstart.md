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

Install other optional extras only when that integration is needed:

```bash
pip install "robometrics[mcap]"      # optional MCAP JSON-message adapter
pip install "robometrics[loggers]"   # optional W&B and MLflow logging
```

For a notebook path, open the
[Colab demo](https://colab.research.google.com/github/gagandeepreehal/robometrics/blob/main/notebooks/robometrics_colab_demo.ipynb).

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
  --threshold fde=1.0 \
  --output result.json
```

The `robometrics` console script is also installed, but `python -m robometrics`
is more reliable in user-level installs where console scripts may be outside
`PATH`.

## Input Shapes

For the full shape, unit, return-type, JSON schema, and logging contract, see
[Input And Output Contract](input_output_contract.md).

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

## Runnable Examples

The repository includes small scripts that mirror the README examples and are
useful smoke checks when developing locally:

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
[Input And Output Contract](input_output_contract.md)

Confirm supported shapes, units, return types, JSON schema behavior, and logger helpers.
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

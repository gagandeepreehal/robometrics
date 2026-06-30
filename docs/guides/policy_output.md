# Getting Started From Policy Output

This guide starts from the shape most robotics users already have: a trained
policy and a rollout log. The goal is to turn that rollout into two small files,
run RoboMetrics, and keep the JSON/report artifacts for CI or experiment review.

## Export One Trajectory Pair

RoboMetrics does not run Isaac Sim, MuJoCo, LeRobot, or a policy checkpoint. Run
your policy in the simulator or evaluation harness you already use, then export
the policy trajectory and the reference trajectory as CSV or JSON.

CSV requires `x` and `y`, with optional `z`:

```csv
t,x,y,z
0.00,0.0,0.0,0.0
0.10,0.4,0.0,0.0
0.20,0.8,0.1,0.0
```

JSON can use `points`:

```json
{
  "points": [[0.0, 0.0], [0.4, 0.0], [0.8, 0.1]]
}
```

Typical mappings:

| Source | Export as |
| --- | --- |
| Isaac Sim policy rollout | End-effector, base, or actor position samples as CSV/JSON |
| MuJoCo rollout | `qpos`-derived body or end-effector XY/XYZ trajectory as CSV/JSON |
| LeRobot episode | Small JSON export with `steps`, `states`, `points`, or `trajectory` |
| ROS 2 bag | JSON message export consumed by `get_adapter("ros2-json")` |

## Validate Files

```bash
pip install "robometrics[io]"
python -m robometrics validate policy_rollout.csv
python -m robometrics validate reference_rollout.csv
```

Validation catches missing `x`/`y`, NaN/inf values, empty trajectories,
inconsistent dimensions, and unsupported file extensions before metrics run.

## Evaluate The Policy

```bash
python -m robometrics evaluate \
  --pred policy_rollout.csv \
  --gt reference_rollout.csv \
  --metrics ade fde hausdorff_distance \
  --threshold ade=0.25 \
  --threshold fde=0.5 \
  --output policy_eval.json
```

The output is strict `EvaluationResult` JSON with `"schema_version": "1"`.
Commit the command and thresholds in your project so future checkpoints are
compared against the same contract.

## Generate A Human Report

```bash
python -m robometrics report policy_eval.json --output policy_report.html
```

Use `policy_eval.json` for CI and experiment tracking. Use `policy_report.html`
for quick review in pull requests, release notes, or lab notebooks.

## Log To Experiment Tools

```python
import wandb

from robometrics import EvaluationResult

run = wandb.init(project="policy-regression")
result = EvaluationResult.from_json(open("policy_eval.json", encoding="utf-8").read())
result.log_to_wandb(run)
```

The same result object also supports `result.log_to_mlflow(...)`. Install
`robometrics[wandb]`, `robometrics[mlflow]`, or `robometrics[loggers]` only when
you want RoboMetrics to import those packages directly.

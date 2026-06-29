# Benchmark Profiles

Benchmark profiles are repeatable local metric sets for smoke tests and CI
regression checks. They are intentionally small and transparent.

```python
from robometrics import list_profiles, run_profile

for profile in list_profiles():
    print(profile.name, profile.metrics)

result = run_profile(
    "policy_regression_ci",
    prediction=[[0.0, 0.0], [1.0, 0.0]],
    ground_truth=[[0.0, 0.0], [1.0, 0.0]],
)
```

## Built-In Profiles

| Profile | Metrics | Purpose |
| --- | --- | --- |
| `trajectory_prediction_basic` | `ade`, `fde`, `hausdorff_distance` | Basic trajectory prediction regression |
| `mobile_robot_safety` | `ade`, `fde`, `lateral_error`, `longitudinal_error` | Geometric path-tracking proxy gate |
| `manipulation_tracking` | `end_effector_tracking_error` | End-effector target tracking gate |
| `policy_regression_ci` | `ade`, `fde` | Minimal deterministic CI profile |

Each profile defines a metric set, default thresholds, expected input schema,
description, and limitations. Profile outputs are normal `EvaluationResult`
JSON files, so they work with `robometrics report` and `robometrics compare`.

Profiles are not benchmark governance. They are starter gates for local
projects, and teams should document any project-specific thresholds beside the
fixture data that justifies them.

## Adding A Profile

Add a `BenchmarkProfile` in `robometrics.benchmarks`, include metric names that
can run from the declared input schema, document limitations, and add tests with
tiny synthetic trajectories.

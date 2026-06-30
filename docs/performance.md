# Performance

RoboMetrics keeps safety metrics NumPy-first so they remain practical in local
CI. `lane_departure_rate()` and `offroad_rate()` use vectorized polygon
containment through `points_in_polygon()`. `collision_rate_obb()` evaluates all
timesteps for each actor with a batched separating-axis test instead of a Python
loop per timestep.

For repeatable JSON timing baselines, run:

```bash
python scripts/benchmark_timing_baselines.py --repeat 3 --output timing_baseline.json
```

The output contains `schema_version`, `robometrics_version`, per-case run
durations, best time, mean time, and deterministic case metadata. Commit a
baseline only when the machine class and workload size are meaningful for your
project; timing values are expected to vary across laptops and CI runners.

Use this smoke benchmark when changing geometry kernels:

```python
import time
import numpy as np

from robometrics import collision_rate_obb, lane_departure_rate

n = 10_000
ego = np.column_stack([np.linspace(0.0, 100.0, n), np.zeros(n)])
lane = np.array([[-1.0, -5.0], [101.0, -5.0], [101.0, 5.0], [-1.0, 5.0]])

start = time.perf_counter()
lane_departure_rate(ego, lane)
print("lane_departure_rate", time.perf_counter() - start)

actors = [ego + np.array([0.2, 0.0]) for _ in range(10)]
yaws = np.zeros(n)
dims = np.array([4.5, 2.0])

start = time.perf_counter()
collision_rate_obb(
    ego_traj=ego,
    ego_dims=dims,
    ego_yaws=yaws,
    actor_trajs=actors,
    actor_dims=[dims for _ in actors],
    actor_yaws=[yaws for _ in actors],
)
print("collision_rate_obb", time.perf_counter() - start)
```

The launch target is that 10 Hz, 10-second scenario checks with 10 actors stay
comfortably below one second on a typical developer laptop.

## Dataset-Scale Driving

Use this benchmark when changing evaluator aggregation, prediction metrics, or
driving safety metrics. It keeps the data synthetic and deterministic while
exercising the same path a CI smoke gate would use.

```python
import time
import numpy as np

from robometrics import Evaluator

rng = np.random.default_rng(7)
samples = 200
timesteps = 20
base = np.stack(
    [
        np.column_stack(
            [
                np.linspace(0.0, 20.0, timesteps),
                np.full(timesteps, lane),
            ]
        )
        for lane in np.linspace(-1.0, 1.0, samples)
    ]
)
ground_truths = [trajectory for trajectory in base]
predictions = [
    trajectory + rng.normal(scale=0.15, size=trajectory.shape)
    for trajectory in ground_truths
]

start = time.perf_counter()
result = Evaluator().evaluate_dataset(
    predictions=predictions,
    ground_truths=ground_truths,
    metrics=["ade", "fde"],
    thresholds={"ade": 0.5, "fde": 1.0},
    bootstrap_ci=1000,
)
print("dataset_driving", time.perf_counter() - start)
print(result.summary())
```

For release checks, the useful signal is trend and order of magnitude. Keep the
same sample count and random seed when comparing two commits.

## Manipulation Workloads

Manipulation metrics are scalar and vectorized, so a dataset-scale check should
batch many short episodes through `evaluate_dataset()` or through direct metric
calls in a loop. The example below exercises end-effector tracking and
force/joint-limit style signals without requiring a simulator.

```python
import time
import numpy as np

from robometrics import (
    contact_richness,
    end_effector_tracking_error,
    force_limit_compliance,
    joint_limit_violation_rate,
)

rng = np.random.default_rng(11)
episodes = 500
timesteps = 30
lower_limits = np.array([-1.0, -1.0, -1.0])
upper_limits = np.array([1.0, 1.0, 1.0])

start = time.perf_counter()
errors = []
compliance = []
violations = []
contacts = []
for _ in range(episodes):
    target = rng.normal(scale=0.2, size=(timesteps, 3)).cumsum(axis=0)
    executed = target + rng.normal(scale=0.03, size=target.shape)
    forces = rng.normal(scale=0.4, size=(timesteps, 3))
    joints = rng.uniform(-1.1, 1.1, size=(timesteps, 3))

    errors.append(end_effector_tracking_error(executed, target))
    compliance.append(force_limit_compliance(forces, max_force=1.5))
    violations.append(joint_limit_violation_rate(joints, lower_limits, upper_limits))
    contacts.append(contact_richness(forces, threshold=0.1))

print("manipulation_dataset", time.perf_counter() - start)
print("mean_ee_error", float(np.mean(errors)))
print("mean_force_compliance", float(np.mean(compliance)))
print("mean_joint_violation_rate", float(np.mean(violations)))
print("mean_contact_richness", float(np.mean(contacts)))
```

These benchmarks are documentation smoke tests, not hard performance contracts.
Use them to catch accidental algorithmic regressions before adding stricter
machine-specific timing gates.

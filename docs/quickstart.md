# Quickstart

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

All trajectory inputs should be finite, non-empty `Nx2` or `Nx3` arrays. Prediction inputs should be `KxTx2` or `KxTx3` arrays.
Single-point trajectories are accepted by metrics with well-defined degenerate outputs, and `Nx3` trajectories are accepted wherever trajectory metrics accept `Nx2`.

Use `metrics="all"` to run every compatible registered metric, or select groups with `categories=["trajectory"]`.
Use `python -m robometrics --help` or `robometrics list-metrics` to inspect an
installed package from the shell.
If `robometrics` is not on `PATH` after a user-level pip install, keep using
`python -m robometrics` or add the script directory reported by pip to `PATH`.

File IO helpers raise `TrajectoryIOError` for missing or malformed files, and `ValueError` for invalid trajectory contents.

# Quickstart

```python
import numpy as np

from robometrics import Evaluator, average_displacement_error, jerk_cost, collision_rate

pred = np.array([[0, 0], [1, 0], [2, 0]])
gt = np.array([[0, 0], [1.1, 0], [2.1, 0]])

ade = average_displacement_error(pred, gt)
comfort = jerk_cost(pred, dt=0.1)
collisions = collision_rate(pred, [], ego_radius=0.5, actor_radius=0.5)

evaluator = Evaluator()
result = evaluator.evaluate(
    prediction=pred,
    ground_truth=gt,
    metrics=["ade", "fde"],
    thresholds={"ade": 1.0},
)

print(result.summary())
print(result.to_markdown())
```

All trajectory inputs should be finite, non-empty `Nx2` or `Nx3` arrays. Prediction inputs should be `KxTx2` or `KxTx3` arrays.

Use `metrics="all"` to run every compatible registered metric, or select groups with `categories=["trajectory"]`.

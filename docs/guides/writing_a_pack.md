# Writing A Pack

Metric packs let teams keep domain-specific metrics outside the core RoboMetrics package while still registering them through the same evaluator and CLI discovery surfaces. A pack is a normal importable Python module with a top-level `METRIC_PACK` list. Each entry describes one metric using the registry contract: name, callable, category, unit, required inputs, optional aliases, optional default keyword arguments, reference, and whether the metric is novel. `load_pack()` imports the module and registers every definition in the global registry.

Keep pack metrics small and deterministic. They should accept NumPy-like arrays or simple Python values, validate their inputs, and return a scalar float or a `MetricResult`. Use `required_inputs` to tell `Evaluator` what keyword inputs must be present. If a metric needs several optional parameters, put stable defaults in `default_kwargs` so CLI and evaluator behavior stays reproducible. Always include a reference string, even when the metric is internal; that metadata appears in list and describe output.

```python
# my_robot_metrics.py
from __future__ import annotations

import numpy as np


def final_x_error(prediction, ground_truth) -> float:
    pred = np.asarray(prediction, dtype=float)
    gt = np.asarray(ground_truth, dtype=float)
    if pred.ndim != 2 or gt.ndim != 2:
        raise ValueError("prediction and ground_truth must be trajectories")
    return float(abs(pred[-1, 0] - gt[-1, 0]))


METRIC_PACK = [
    {
        "name": "final_x_error",
        "fn": final_x_error,
        "category": "trajectory",
        "unit": "meters",
        "required_inputs": ("prediction", "ground_truth"),
        "reference": "Team internal longitudinal endpoint error",
        "is_novel": True,
    }
]
```

Register and use the pack like this:

```python
import numpy as np
from robometrics import Evaluator, load_pack

load_pack("my_robot_metrics")
result = Evaluator().evaluate(
    prediction=np.array([[0.0, 0.0], [1.2, 0.0]]),
    ground_truth=np.array([[0.0, 0.0], [1.0, 0.0]]),
    metrics=["final_x_error"],
)
print(result.to_markdown())
```

During development, unregister test metrics or run pack tests in a fresh process so global registry state does not leak between tests. For published packs, document the expected inputs and units beside the module.

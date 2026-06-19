# RoboMetrics

RoboMetrics is a lightweight Python library for evaluating robotics and autonomy outputs with typed, local metric functions. It covers trajectories, multi-modal prediction, driving safety, comfort, physics feasibility, task outcomes, manipulation signals, calibration, coverage, and experiment comparison without requiring a simulator, dashboard, ROS install, or cloud service.

## Installation

```bash
pip install robometrics
```

Install optional I/O support when you want CSV helpers that use pandas:

```bash
pip install "robometrics[io]"
```

## 30-Second Quickstart

```python
import numpy as np
from robometrics import Evaluator

predictions = [
    np.array([[0.0, 0.0], [1.1, 0.0], [2.0, 0.0]]),
    np.array([[0.0, 0.0], [1.3, 0.0], [2.5, 0.0]]),
]
ground_truths = [
    np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
    np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
]

result = Evaluator().evaluate_dataset(
    predictions=predictions,
    ground_truths=ground_truths,
    metrics=["ade", "fde"],
    thresholds={"ade": 0.5, "fde": 1.0},
    bootstrap_ci=1000,
)

print(result.to_markdown())
```

## Documentation Map

- `docs/metrics/` documents metric families with units, formulas, references, and directionality.
- `docs/guides/comparing_policies.md` shows policy checkpoint comparison with `EvaluationResult.compare()`.
- `docs/guides/ci_integration.md` shows how to wire `robometrics compare` into GitHub Actions.
- `docs/guides/writing_a_pack.md` shows third-party metric pack registration.
- `docs/guides/ros_adapter.md` shows ROS message conversion when ROS is installed and sourced.
- `docs/api.md` lists the public symbols exported from `robometrics.__all__`.

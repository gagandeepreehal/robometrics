# Comparing Policies

Policy comparison is useful when you have two checkpoints, two planner versions, or a baseline and candidate policy that were evaluated on the same scenario set. RoboMetrics stores metric outputs in `EvaluationResult` objects, so comparison can stay local and version-controlled. The comparison API matches metric names, computes the delta from A to B, estimates percent change when the A value is finite and nonzero, and marks the winner for each metric. Lower is treated as better for most error metrics, while score, accuracy, diversity, smoothness, and selected success-rate style names are treated as higher-is-better.

The most common workflow is to save one JSON file for the baseline and one JSON file for the candidate. You can then compare in Python for notebooks or reports, and use the CLI for CI. Keep the input dataset, metric list, thresholds, and sampling seed stable across both runs; otherwise the comparison may reflect evaluation drift rather than policy quality. For metrics that have thresholds, `robometrics compare` exits with status code 0 only when B wins on all thresholded metrics.

```python
import numpy as np
from robometrics import Evaluator

gt = [np.array([[0.0, 0.0], [1.0, 0.0]])]
baseline = [np.array([[0.0, 0.0], [1.6, 0.0]])]
candidate = [np.array([[0.0, 0.0], [1.1, 0.0]])]

evaluator = Evaluator()
result_a = evaluator.evaluate_dataset(
    predictions=baseline,
    ground_truths=gt,
    metrics=["ade", "fde"],
    thresholds={"ade": 0.5, "fde": 0.75},
)
result_b = evaluator.evaluate_dataset(
    predictions=candidate,
    ground_truths=gt,
    metrics=["ade", "fde"],
    thresholds={"ade": 0.5, "fde": 0.75},
)

comparison = result_a.compare(result_b)
print(comparison.to_markdown())
```

The comparison table is intentionally small: it reports metric, A value, B value, delta, and winner. For richer reports, store the original `EvaluationResult` JSON beside the comparison so reviewers can inspect metadata such as category, unit, threshold, confidence intervals, and dataset sample count. When a metric is missing from one side, the comparison keeps the metric but uses `nan`, which makes the winner a tie. Treat that as an evaluation setup problem and fix the metric selection before using the result for release decisions.

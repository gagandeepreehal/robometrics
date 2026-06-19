# CI Integration

Continuous integration is a good fit for guardrail metrics: collision rate, lane departure rate, offroad rate, miss rate, final displacement error, and any task metric with a clear acceptance threshold. The recommended pattern is to keep a baseline JSON artifact in the repository or download it from a stable artifact store, run the candidate evaluation in CI, then call `robometrics compare baseline.json new.json`. The command prints a comparison and exits with status code 0 only when the candidate result B wins on all thresholded metrics. That makes the shell step fail naturally when a thresholded safety metric regresses.

Use stable evaluation data in CI. If the dataset is too large, run a small deterministic scenario suite as a smoke gate and reserve full leaderboard or nightly runs for heavier workflows. Store the exact metric list and thresholds in code, not in free-form job comments. When you add `bootstrap_ci` in dataset evaluation, remember that CI still compares point estimates; the interval is metadata for human review unless you encode a separate acceptance rule.

```yaml
name: robometrics

on:
  pull_request:

jobs:
  compare-policy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install .
      - run: python scripts/evaluate_policy.py --output new.json
      - run: robometrics compare baseline.json new.json --format markdown
```

The evaluation script should write an `EvaluationResult` JSON file. A minimal local version looks like this:

```python
from pathlib import Path
import numpy as np
from robometrics import Evaluator

result = Evaluator().evaluate_dataset(
    predictions=[np.array([[0.0, 0.0], [1.1, 0.0]])],
    ground_truths=[np.array([[0.0, 0.0], [1.0, 0.0]])],
    metrics=["ade", "fde"],
    thresholds={"ade": 0.5, "fde": 0.75},
)
Path("new.json").write_text(result.to_json(), encoding="utf-8")
```

For safety-focused repositories, include `collision_rate`, `collision_rate_obb`, `lane_departure_rate`, or `offroad_rate` when the required inputs are available. Keep the baseline file reviewed and intentional; changing it should be treated like changing a test expectation.

# CI Integration

Continuous integration is a good fit for guardrail metrics: collision rate, lane departure rate, offroad rate, miss rate, final displacement error, and any task metric with a clear acceptance threshold. The recommended pattern is to keep a baseline JSON artifact in the repository or download it from a stable artifact store, run the candidate evaluation in CI, then call `python -m robometrics compare baseline.json new.json`. The command prints a comparison and exits with status code 0 when candidate result B wins or exactly ties every thresholded metric. A thresholded metric that regresses, is missing from B, or has a non-finite value makes the shell step fail naturally.

Evaluator thresholds follow metric direction. Lower-is-better metrics pass when
`value <= threshold`; higher-is-better metrics pass when `value >= threshold`.
Keep this in mind for score-like metrics such as `task_success_rate` or
`workspace_coverage`.

Use stable evaluation data in CI. If the dataset is too large, run a small deterministic scenario suite as a smoke gate and reserve full leaderboard or nightly runs for heavier workflows. Store the exact metric list and thresholds in code, not in free-form job comments. When you add `bootstrap_ci` in dataset evaluation, remember that CI still compares point estimates; the interval is metadata for human review unless you encode a separate acceptance rule.

For file-based smoke gates, keep tiny CSV/JSON fixtures in the repository and
use the CLI directly:

```yaml
- name: Evaluate regression fixture
  run: |
    python -m robometrics evaluate \
      --pred examples/fixtures/predictions.csv \
      --gt examples/fixtures/ground_truth.csv \
      --metrics ade fde \
      --threshold ade=0.5 \
      --threshold fde=1.0 \
      --output result.json

- name: Compare with baseline
  run: python -m robometrics compare examples/fixtures/baseline_result.json result.json

- name: Generate report
  run: python -m robometrics report result.json --output report.html
```

Upload `report.html` as a CI artifact when you want a human-readable summary.

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
      - run: python -m robometrics compare baseline.json new.json --format markdown
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

The first time you run this, save the output as `baseline.json` and commit it to
the repository. Subsequent CI runs compare against that committed file.

> **Tip — JSON schema:** `EvaluationResult.to_json()` produces an object with
> `"schema_version": "1"` and a `"results"` list of metric records. Do not
> hand-craft this file; always generate it via `to_json()`. Loading a file with
> the wrong key (e.g. `"metrics"` instead of `"results"`) raises a `ValueError`;
> loading a file with an unknown `schema_version` also raises. See the
> [result schema migration guide](result_schema_migration.md) for the versioning
> policy.

```python
# One-time: generate and commit baseline.json
result = Evaluator().evaluate_dataset(...)
Path("baseline.json").write_text(result.to_json(), encoding="utf-8")
```

For safety-focused repositories, include `collision_rate`, `collision_rate_obb`, `lane_departure_rate`, or `offroad_rate` when the required inputs are available. Keep the baseline file reviewed and intentional; changing it should be treated like changing a test expectation.

`robometrics history <directory>` only reads files named `*_eval.json` or
`*.eval.json`. Use names such as `checkpoint_100.eval.json` when you want a
directory of evaluation results to appear in the history summary.

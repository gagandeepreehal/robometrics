# Contributing to RoboMetrics

Thanks for helping improve RoboMetrics. The project should stay small, local-first, typed, and simulator-independent.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Quality Checks

Run these before opening a pull request:

```bash
ruff check .
mypy robometrics
pytest --cov=robometrics --cov-report=term-missing
python examples/basic_metrics.py
python examples/evaluator_quickstart.py
python examples/thresholds.py
python examples/export_results.py
python examples/trajectory_metrics.py
python examples/prediction_metrics.py
python examples/driving_metrics.py
python examples/safety_metrics.py
python examples/comfort_metrics.py
python examples/load_from_csv.py
python examples/evaluator_usage.py
python examples/manipulation_metrics.py
```

## Adding Metrics

- Keep metric functions deterministic and free of simulator, ROS, cloud, or GPU dependencies.
- Validate input shape, dimensionality, and non-finite values.
- Document units and return types.
- Add tests for normal cases and edge cases.
- Register new public metrics in the default registry when they are stable enough for evaluator use.

## Pull Requests

Include a concise summary, tests run, and any compatibility concerns. Avoid unrelated formatting churn or broad refactors in metric changes.

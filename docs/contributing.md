# Contributing

See the root [CONTRIBUTING.md](../CONTRIBUTING.md) for the full contributor guide.

## Local Checks

```bash
ruff check .
mypy robometrics
pytest --cov=robometrics --cov-report=term-missing
python examples/basic_metrics.py
python examples/evaluator_quickstart.py
python examples/thresholds.py
python examples/export_results.py
```

## Contribution Scope

RoboMetrics is intended to remain a lightweight local Python package. New contributions should avoid adding dashboards, services, databases, simulator dependencies, ROS dependencies, or cloud requirements unless the project explicitly changes scope.

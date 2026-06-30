# Contributing

See the repository [CONTRIBUTING.md](https://github.com/gagandeepreehal/robometrics/blob/main/CONTRIBUTING.md) for the full contributor guide.

## Local Checks

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
python examples/new_metrics_example.py
python examples/load_from_csv.py
python examples/dataset_loader_evaluation.py
python examples/evaluator_usage.py
python examples/manipulation_metrics.py
```

## Contribution Scope

RoboMetrics is intended to remain a lightweight local Python package. New contributions should stay focused on typed metric functions, local file IO, tests, and documentation.

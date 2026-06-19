# Roadmap And Stability

RoboMetrics `0.x` releases are public alpha releases. Metric function names,
units, input shapes, result JSON fields, and registry metadata are intended to
stay stable within a minor release. Breaking changes before `1.0.0` will be
called out in `CHANGELOG.md` and should include migration notes.

## Stable In 0.1.x

- Direct NumPy metric functions for trajectory, prediction, safety, comfort,
  physics, task, manipulation, calibration, coverage, diversity, and temporal
  checks.
- `Evaluator`, `MetricRegistry`, `MetricDefinition`, `MetricResult`, and
  `EvaluationResult` data contracts.
- Standards-compliant JSON export for non-finite metric values.
- CLI metric discovery, comparison, and history summaries.

## Planned For 0.2.0

- Publish the documentation site from `docs/` through GitHub Pages.
- Expand performance benchmarks for dataset-scale driving and manipulation
  workloads.
- Add more worked examples for probabilistic prediction, custom packs, and
  history summaries.
- Review threshold semantics for higher-is-better metrics before declaring the
  evaluator API beta-stable.

## Before 1.0.0

- Keep CI green across Python 3.9 through 3.12.
- Keep test coverage at or above 90%.
- Preserve documented metric directionality for comparison workflows.
- Add migration notes for any public API changes.

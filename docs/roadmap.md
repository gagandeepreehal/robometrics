# Roadmap And Stability

RoboMetrics `0.x` releases are public alpha releases. Metric function names,
units, input shapes, result JSON fields, and registry metadata are intended to
stay stable within a minor release. Breaking changes before `1.0.0` will be
called out in `CHANGELOG.md` and should include migration notes.

## Stable In 0.2.x

- Direct NumPy metric functions for trajectory, prediction, safety, comfort,
  physics, task, manipulation, calibration, coverage, diversity, and temporal
  checks.
- `Evaluator`, `MetricRegistry`, `MetricDefinition`, `MetricResult`, and
  `EvaluationResult` data contracts.
- Standards-compliant JSON export for non-finite metric values.
- CLI metric discovery, comparison, and history summaries.
- Documentation publishing through GitHub Pages.
- Dataset-scale performance guidance for driving and manipulation workloads.
- Worked examples for probabilistic prediction, custom packs, and history
  summaries.
- Direction-aware evaluator thresholds. Metrics marked `higher_is_better=True`
  pass when `value >= threshold`; other metrics pass when `value <= threshold`.

## Planned For 0.3.0

- Add small benchmark scripts that can emit repeatable JSON timing baselines.
- Expand loader examples for directory-level and dataset-level evaluation.
- Add migration guidance if any public result schema fields change.

## Before 1.0.0

- Keep CI green across Python 3.9 through 3.12.
- Keep test coverage at or above 90%.
- Preserve documented metric directionality for comparison workflows.
- Add migration notes for any public API changes.

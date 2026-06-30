# Roadmap And Stability

RoboMetrics `0.x` releases are public alpha releases. Metric function names,
units, input shapes, result JSON fields, and registry metadata are intended to
stay stable within a minor release. Breaking changes before `1.0.0` will be
called out in `CHANGELOG.md` and should include migration notes.

## Stable In 0.3.x

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
- `scripts/benchmark_timing_baselines.py` emits repeatable JSON timing baselines
  for geometry safety, dataset evaluator, and manipulation metric workloads.
- `examples/dataset_loader_evaluation.py` and the loader examples guide cover
  directory-level loading and matched dataset evaluation.
- The result schema migration guide documents the version-1 JSON contract and
  the required migration shape for any future result schema change.

## Before 1.0.0

- Keep CI green across Python 3.9 through 3.12.
- Keep test coverage at or above 90%.
- Preserve documented metric directionality for comparison workflows.
- Add migration notes for any public API changes.

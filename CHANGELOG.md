# Changelog

All notable changes to RoboMetrics will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project follows semantic versioning once public releases begin.

## [Unreleased]

## [0.1.2] - 2026-06-20

### Fixed

- Corrected comparison directionality for `offroad_rate` and
  `joint_limit_violation_rate`.
- Made `robometrics compare` pass exact ties on thresholded metrics while still
  failing missing or non-finite candidate values.
- Loaded metric packs into a caller-provided registry with
  `load_pack(module_name, registry=custom_registry)`.
- Vectorized polygon containment and OBB collision checks used by safety
  metrics.
- `_error_result` now sets `passed=None` instead of `passed=False` so that
  `strict_passed` correctly ignores runtime errors on metrics that have no
  threshold assigned.

### Added

- Registry-level `higher_is_better` metadata for built-in and custom metrics.
- Documentation site configuration, performance notes, and alpha roadmap.
- CI coverage threshold and a Python 3.12 `[io]` extras matrix cell.
- `Evaluator` support for `Trajectory` schema inputs and configurable
  `bootstrap_seed` for dataset confidence intervals.

## [0.1.1] - 2026-06-19

### Added

- NumPy-based trajectory, prediction, comfort, safety, and physics metrics.
- Local `Evaluator`, metric registry, structured results, thresholds, and exporters.
- CSV, JSON, NumPy, and NPZ trajectory loading helpers.
- Release-readiness documentation, examples, CI, and contribution files.

### Notes

- This is a public alpha candidate, not production-proven software.

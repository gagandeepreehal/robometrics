# CLI Commands

RoboMetrics CLI commands are local-first wrappers around the Python API. They
read small CSV/JSON files, emit strict JSON, and avoid heavyweight services.
Examples use `python -m robometrics` so they work even when a user-level pip
install places console scripts outside `PATH`; the `robometrics` executable is
equivalent when it is on `PATH`.

## Evaluate

```bash
cat > predictions.csv <<'CSV'
t,x,y
0,0.0,0.0
1,1.0,0.0
2,2.0,0.0
CSV

cat > ground_truth.csv <<'CSV'
t,x,y
0,0.0,0.0
1,1.1,0.0
2,2.1,0.0
CSV

python -m robometrics evaluate \
  --pred predictions.csv \
  --gt ground_truth.csv \
  --metrics ade fde \
  --threshold ade=0.5 \
  --threshold fde=1.0 \
  --output result.json
```

`evaluate` validates that both files exist, loads CSV or JSON trajectory data
through the existing IO helpers, runs the requested metrics using each metric's
registry compatibility rules, and writes an `EvaluationResult` JSON file.
The file includes top-level `"schema_version": "1"` as the stable contract for
CI parsers and downstream tools.
Aligned-sample metrics such as `ade` and `fde` require matching prediction and
ground-truth shapes. Set-based metrics such as `hausdorff_distance` can compare
different numbers of points when coordinate dimensionality matches.
Thresholds use the same direction-aware contract as `Evaluator`: lower-is-better
metrics pass with `value <= threshold`, and higher-is-better metrics pass with
`value >= threshold`.

## Validate

```bash
python -m robometrics validate predictions.csv
python -m robometrics validate predictions.csv --output validation.json
```

`validate` inspects trajectory-style CSV/JSON files for missing required
fields, invalid numeric values, NaN/inf, inconsistent 2D/3D dimensions,
non-monotonic timestamps when timestamp columns exist, empty trajectories, and
unsupported formats. It prints a human-readable summary and can also write a
JSON validation report.

## Report

```bash
python -m robometrics report result.json --output report.html
```

`report` creates a standalone static HTML file with a summary table, metric
values, pass/fail indicators, metadata, and a baseline comparison section when
the result payload contains comparison metadata.

## Benchmark Profiles

```bash
python -m robometrics benchmark list
python -m robometrics benchmark run policy_regression_ci \
  --pred predictions.csv \
  --gt ground_truth.csv \
  --output result.json
```

Profiles are small smoke/regression definitions, not leaderboard definitions.
They package metric lists, default thresholds, expected input schemas,
descriptions, and limitations.

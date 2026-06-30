# Input And Output Contract

This page mirrors the public contract summarized in the README: what shapes
RoboMetrics accepts, what units it assumes, and what result objects return.

## Input Shapes

Trajectory-like inputs are NumPy-compatible arrays with finite numeric values:

| Input | Shape | Notes |
| --- | --- | --- |
| Single trajectory | `Nx2` or `Nx3` | `N` timesteps with XY or XYZ coordinates. |
| Multimodal predictions | `KxTx2` or `KxTx3` | `K` ranked modes, `T` timesteps. |
| Prediction ground truth | `Tx2` or `Tx3` | One reference path for prediction metrics. |
| Actor trajectories | list of `Nx2` or `Nx3` arrays | Time-aligned actor paths for safety metrics. |
| Time-series metrics | `TxD` or `BxTxD` | Temporal, rollout, action, and control metrics. |
| Coverage samples or behavior embeddings | `NxD` | State, action, workspace, or embedding samples. |
| Batched behavior trajectories or actions | `NxTxD` | Dataset-style behavior arrays. |

`N` or `T` is the number of timesteps, `K` is the number of prediction modes,
and position columns are in meters unless a metric says otherwise.

`Nx3` inputs are supported by metrics that operate on positions. ADE, FDE,
Hausdorff distance, prediction distance metrics, and path length use all
coordinate dimensions. Curvature, lateral error, longitudinal error, collision
checks, lane departure, and constant-velocity TTC are planar XY metrics.

Empty arrays, `NaN` or infinite values, bad ranks, invalid `dt`, and
incompatible dimensions raise `ValueError` with targeted messages. Metrics that
compare aligned samples, such as ADE and FDE, require matching trajectory
shapes. Set-based metrics such as `hausdorff_distance` may compare different
numbers of points when coordinate dimensionality matches.

The evaluator also accepts `Trajectory` schema objects and converts them with
`.array()` before dispatching metric functions. PyTorch-style tensor objects
with `.detach().cpu().numpy()` are accepted without making PyTorch a
RoboMetrics dependency.

## Units

| Quantity | Unit |
| --- | --- |
| Position and distance inputs | meters |
| Time step `dt` | seconds |
| Speed | meters per second |
| Acceleration | meters per second squared |
| Jerk | meters per second cubed |
| Curvature | inverse meters |
| Rates and scores | unitless floats |

Some metrics intentionally return context-dependent quantities. For example,
`path_length()` reports total traveled distance, while `speed_profile()` reports
one speed estimate per input point. Endpoint speeds are finite-difference
gradient estimates, not `N-1` interval speeds.

## Return Types

Most public metric functions return `float`. Profile-style functions such as
`curvature()`, `speed_profile()`, `acceleration()`, and `jerk()` return per-step
NumPy arrays.

For scalar acceleration or jerk summaries, use helpers such as:

- `acceleration_magnitude()`
- `jerk_magnitude()`
- `mean_acceleration()`
- `rms_acceleration()`
- `jerk_cost()`

Threshold-style physics helpers return `MetricResult`, which stores:

- `value`
- `unit`
- `threshold`
- `passed`
- `metadata`

`EvaluationResult` is the container returned by `Evaluator`. It can export and
reload dictionaries, strict JSON, Markdown tables, CSV, and pandas DataFrames.
Install `robometrics[io]` for CSV and pandas-backed exports.

## Result JSON

Evaluation JSON includes top-level `"schema_version": "1"`. Downstream CI,
dashboards, and experiment trackers should treat this as the stable output
contract and reject unknown future schema versions instead of guessing.

Non-finite metric values are serialized as `null` in JSON with metadata that
records whether the original value was `nan`, `inf`, or `-inf`.

Use the library reader rather than hand-parsing result files:

```python
from robometrics import EvaluationResult

result = EvaluationResult.from_json(path.read_text(encoding="utf-8"))
```

See [Result Schema Migration](guides/result_schema_migration.md) for the
versioning and reader policy.

## Experiment Logging

Use `result.log_to_wandb(run)` or `result.log_to_mlflow(run)` to send finite
metric values and summary counts into existing experiment runs:

```python
result.log_to_wandb(wandb_run)
result.log_to_mlflow(mlflow)
```

Passing a run or logger object avoids importing optional logging packages.
Install `robometrics[wandb]`, `robometrics[mlflow]`, or
`robometrics[loggers]` when you want RoboMetrics to import those tools
directly.

## CSV And JSON Files

```python
from robometrics import load_trajectory_csv, load_trajectory_json

csv_traj = load_trajectory_csv("trajectory.csv")
json_traj = load_trajectory_json("trajectory.json")
```

CSV files must include `x` and `y` columns and may include `z`. The generic CSV
adapter preserves a present `z` column when constructing `Trajectory` objects.
JSON files may contain either a raw list of points or an object with a `points`
field.

Use `load_trajectory_dir()` to load every supported trajectory file in a
directory into a filename-keyed dictionary.

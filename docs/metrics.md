# Metric Catalog

Use this page to choose metric families and confirm the direction of each
metric before wiring thresholds into an evaluator or CI gate. Built-in metrics
are registered through `robometrics.registry`; common displacement metrics use
short canonical names such as `ade` and `fde`, while aliases remain available.

<div class="rm-grid rm-grid-3" markdown="1">
<div class="rm-card" markdown="1">
**Path quality**

[Trajectory](metrics/trajectory.md), [Prediction](metrics/prediction.md), and
[Temporal](metrics/temporal.md) metrics measure tracking error, forecast modes,
and rollout drift.
</div>

<div class="rm-card" markdown="1">
**Safety and feasibility**

[Safety](metrics/safety.md), [Driving](metrics/driving.md), and
[Physics](metrics/physics.md) metrics inspect collisions, road containment,
speed, acceleration, jerk, and feasibility constraints.
</div>

<div class="rm-card" markdown="1">
**Task and dataset signals**

[Task](metrics/task.md), [Manipulation](metrics/manipulation.md),
[Coverage](metrics/coverage.md), [Diversity](metrics/diversity.md), and
[Calibration](metrics/calibration.md) cover completion, contact, exploration,
sample spread, and confidence quality.
</div>
</div>

## Aliases

| Alias | Canonical metric |
| --- | --- |
| `average_displacement_error` | `ade` |
| `final_displacement_error` | `fde` |
| `minade` | `min_ade` |
| `minfde` | `min_fde` |

## Family Summary

| Family | Typical inputs | Common units | Direction |
| --- | --- | --- | --- |
| [Trajectory](metrics/trajectory.md) | `Nx2` or `Nx3` paths | meters, `1/m` | Mostly lower is better |
| [Prediction](metrics/prediction.md) | `KxTx2` or `KxTx3` modes | meters, ratio, nats | Mostly lower is better |
| [Driving](metrics/driving.md) | Ranked predictions, drivable polygons, actors | meters, ratio, seconds | Mixed |
| [Safety](metrics/safety.md) | Ego path, actors, lane/drivable geometry | ratio, meters, seconds | Mixed |
| [Comfort](metrics/comfort.md) | Sampled trajectories and `dt` | `m/s^2`, `m/s^3`, score | Lower magnitude or higher score |
| [Physics](metrics/physics.md) | Trajectories, timestamps, constraints, forces | speed, acceleration, score | Mixed |
| [Task](metrics/task.md) | Outcomes, positions, goals | ratio | Higher is better |
| [Manipulation](metrics/manipulation.md) | Grasps, contact forces, joints, end-effector paths | ratio, force, meters | Mixed |
| [Calibration](metrics/calibration.md) | Confidence and correctness arrays | ratio | Lower is better |
| [Coverage](metrics/coverage.md) | Finite sample matrices | ratio, cells | Higher is better |
| [Diversity](metrics/diversity.md) | Embeddings, trajectories, modes | behavior units, meters | Higher is better |
| [Temporal](metrics/temporal.md) | `TxD` or `BxTxD` rollouts/actions | error/timestep, score | Mixed |

## Registered Metrics

### Trajectory

| Metric | What it reports | Direction |
| --- | --- | --- |
| `ade` / `average_displacement_error` | Mean pointwise Euclidean distance | Lower is better |
| `fde` / `final_displacement_error` | Final-point Euclidean distance | Lower is better |
| `hausdorff_distance` | Worst-case geometric mismatch between point sets | Lower is better |
| `path_length` | Total traveled distance | Context dependent |
| `curvature` / `curvature_profile` | Planar XY curvature profile | Lower is smoother |
| `mean_curvature` | Mean planar XY curvature | Lower is smoother |
| `lateral_error` | Perpendicular deviation from a reference path | Lower is better |
| `longitudinal_error` | Along-track lead or lag from a reference path | Lower is better |

ADE, FDE, Hausdorff distance, prediction distance metrics, and path length use
all provided coordinate dimensions. Curvature, lateral/longitudinal error,
safety geometry, and TTC are planar XY metrics.

### Prediction

| Metric | What it reports | Direction |
| --- | --- | --- |
| `min_ade` | Best ADE across prediction modes | Lower is better |
| `min_fde` | Best FDE across prediction modes | Lower is better |
| `miss_rate` | Whether all modes miss the final-point threshold | Lower is better |
| `topk_trajectory_error` | Best ADE among the first `k` ranked modes | Lower is better |
| `prediction_nll` | Confidence-weighted negative log likelihood | Lower is better |
| `displacement_at_k` | Best displacement error among the first `k` ranked modes | Lower is better |

### Temporal And Robustness

| Metric | What it reports | Direction |
| --- | --- | --- |
| `temporal_drift` | Slope of per-timestep L2 error | Lower is better |
| `action_jerk` | Mean squared second finite difference of actions | Lower is smoother |
| `control_smoothness` | Bounded smoothness score from action jerk | Higher is smoother |
| `long_horizon_drift` | Later-weighted rollout error | Lower is better |
| `compounding_error_index` | Normalized non-negative error growth | Lower is better |

### Comfort And Control

| Metric | What it reports | Direction |
| --- | --- | --- |
| `acceleration` | Per-step acceleration vectors | Lower magnitude is smoother |
| `jerk` | Per-step jerk vectors | Lower magnitude is smoother |
| `acceleration_magnitude` | Per-step acceleration magnitudes | Lower is smoother |
| `jerk_magnitude` | Per-step jerk magnitudes | Lower is smoother |
| `jerk_cost` | Mean squared jerk magnitude | Lower is smoother |
| `max_acceleration` | Maximum acceleration magnitude | Lower is smoother |
| `mean_acceleration` | Mean acceleration magnitude | Lower is smoother |
| `rms_acceleration` | Root-mean-square acceleration magnitude | Lower is smoother |
| `max_deceleration` | Maximum longitudinal deceleration magnitude | Lower is smoother |
| `smoothness_score` | Unitless shape score from normalized third finite difference | Higher is smoother |

`smoothness_score` is spatial-scale-invariant for geometrically similar paths
and separate from physical `jerk_cost(traj, dt)`. Trajectories shorter than four
points return `1.0` with a `RuntimeWarning` because third finite differences
are not measurable.

### Safety And Driving

| Metric | What it reports | Direction |
| --- | --- | --- |
| `collision_rate` | Disc collision fraction across actor-covered timesteps | Lower is better |
| `collision_rate_obb` | Oriented-box collision fraction | Lower is better |
| `time_to_collision` | Constant-velocity disc-agent TTC | Higher is safer |
| `min_distance_to_actors` | Closest time-aligned XY actor distance | Higher is safer |
| `lane_departure_rate` | Fraction of ego points outside a lane polygon | Lower is better |
| `offroad_rate` | Fraction of ego positions outside drivable areas | Lower is better |
| `soft_ttc` | Minimum rollout TTC under local velocity estimates | Higher is safer |
| `recovery_success_rate` | Successful recoveries divided by opportunities | Higher is better |
| `failure_severity` | Mean or max incident severity | Lower is better |
| `near_miss_rate` | Fraction of clearances below threshold, excluding collisions | Lower is better |
| `intervention_free_time` | Longest or mean non-intervention segment duration | Higher is better |

### Coverage, Calibration, Diversity, Task, And Manipulation

| Metric | What it reports | Direction |
| --- | --- | --- |
| `coverage_score` | Occupied finite grid bins divided by total bins | Higher is better |
| `workspace_coverage` | Unique discretized workspace cells visited | Higher is better |
| `calibration_error` | Expected calibration error | Lower is better |
| `behavioral_diversity` | Mean pairwise distance between behavior embeddings | Higher is better |
| `trajectory_diversity` | Mean pairwise ADE between prediction modes | Higher is better |
| `task_success_rate` | Mean binary task success | Higher is better |
| `goal_reaching_accuracy` | Fraction of positions within goal tolerance | Higher is better |
| `grasp_success_rate` | Successful grasps divided by attempts | Higher is better |
| `contact_richness` | Fraction of meaningful contact-force timesteps | Higher is better |
| `force_limit_compliance` | Fraction of force samples inside the limit | Higher is better |
| `joint_limit_violation_rate` | Fraction of configurations outside joint limits | Lower is better |
| `end_effector_tracking_error` | Mean end-effector tracking error | Lower is better |

### Physical Consistency

| Metric | What it reports | Direction |
| --- | --- | --- |
| `speed_profile` | Speed estimate at each trajectory point | Context dependent |
| `acceleration_limits_violated` | Whether acceleration exceeds a limit | Lower is better |
| `jerk_limits_violated` | Whether jerk exceeds a limit | Lower is better |
| `curvature_limits_violated` | Whether curvature exceeds a limit | Lower is better |
| `dynamic_feasibility_score` | Worst-relative-violation feasibility score | Higher is more feasible |
| `kinematic_feasibility` | Velocity, acceleration, and curvature feasibility score | Higher is more feasible |
| `dynamic_feasibility` | Force, acceleration, torque, and friction feasibility score | Higher is more feasible |
| `physics_violation_rate` | Fraction of timesteps or events with any violation | Lower is better |

`dynamic_feasibility_score` supports `max_speed`, `max_accel`, `max_jerk`, and
`max_curvature`. Unknown constraint keys raise `ValueError`.

## Category Taxonomy

The registry categories are intentionally pragmatic. `comfort` contains
kinematic profiles and smoothness values used for ride/control quality,
`temporal` contains rollout drift and compounding-error metrics, and `physics`
contains dynamic limit checks such as `acceleration_limits_violated()` and
`dynamic_feasibility_score()`. Running all kinematic quantities may require
selecting multiple categories or explicit metric names.

## Edge-Case Behavior

| Case | Behavior |
| --- | --- |
| Empty arrays, one-dimensional arrays, invalid shapes, `NaN`, or infinite values | Raise `ValueError` |
| Single-point trajectories | Accepted by metrics with defined degenerate outputs |
| `Nx3` trajectories | Supported wherever trajectory inputs accept `Nx2`; planar metrics use XY |
| Thresholded physics metrics | Return `MetricResult` with `value`, `unit`, `passed`, `threshold`, and `metadata` |
| `min_distance_to_actors(ego_traj, [])` | Returns `math.inf` |
| `recovery_success_rate()` with no opportunities | Returns `nan` |
| `failure_severity([])` | Returns `0.0` |
| `coverage_score()` with out-of-bounds samples | Ignores out-of-bounds samples instead of clipping |
| `physics_violation_rate()` with multiple violation types | Uses logical OR by default |

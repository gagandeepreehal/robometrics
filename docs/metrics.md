# Metrics

Built-in metrics are registered for evaluation through `robometrics.registry`. The evaluator uses short canonical names for common displacement metrics, including `ade` and `fde`, while aliases remain available through registry lookup.

Registered aliases:

- `average_displacement_error` -> `ade`
- `final_displacement_error` -> `fde`
- `minade` -> `min_ade`
- `minfde` -> `min_fde`

## Trajectory

- `average_displacement_error(pred, gt)`: mean pointwise Euclidean distance.
- `final_displacement_error(pred, gt)`: final-point Euclidean distance.
- `hausdorff_distance(pred, gt)`: symmetric Hausdorff distance using all coordinate dimensions.
- `path_length(traj)`: total Euclidean path length.
- `curvature(traj)`: approximate planar XY curvature at each point; returns an array.
- `curvature_profile(traj)`: alias for `curvature(traj)`.
- `mean_curvature(traj)`: mean planar XY curvature as a scalar.
- `lateral_error(pred, ref)`: mean absolute planar XY perpendicular error from a reference path.
- `longitudinal_error(pred, ref)`: mean absolute planar XY along-track error from a reference path.

ADE, FDE, Hausdorff distance, prediction distance metrics, and path length use
all provided coordinate dimensions. Curvature, lateral/longitudinal error,
safety geometry, and TTC are planar XY metrics.

## Prediction

- `min_ade(predictions, gt)`: best average displacement error across modes.
- `min_fde(predictions, gt)`: best final displacement error across modes.
- `miss_rate(predictions, gt, threshold)`: per-sample miss indicator; returns `1.0` when all modes miss the final-point threshold, otherwise `0.0`. Average across samples for a dataset miss rate.
- `topk_trajectory_error(predictions, gt, k)`: best ADE among the first `k` modes, assuming predictions are already ranked by descending confidence.

## Temporal and Robustness

- `temporal_drift(predicted, reference)`: least-squares slope of per-timestep L2 error over timestep index for `TxD` or `BxTxD` arrays. Perfect prediction returns `0.0`; increasing error returns a positive value; decreasing error returns a negative value.
- `action_jerk(actions, dt=1.0)`: mean squared L2 norm of the second finite difference of `TxD` or `BxTxD` action sequences divided by `dt**2`. Lower is smoother. Constant and linear action sequences return `0.0`.
- `control_smoothness(actions, dt=1.0)`: bounded score `1 / (1 + action_jerk(actions, dt))`; higher is smoother and constant/linear controls return `1.0`.
- `long_horizon_drift(predicted, reference)`: later-weighted mean L2 rollout error with weights `1..T`. Perfect rollouts return `0.0`; the same error is penalized more when it appears later in the horizon.
- `compounding_error_index(errors_or_predicted, reference=None, epsilon=1e-12)`: normalized non-negative growth index `max(0, final_error - initial_error) / (mean_error + epsilon)`. It accepts explicit error curves shaped `T` or `BxT`, or predicted/reference state arrays shaped `TxD` or `BxTxD`. `0.0` means error did not compound from first to final timestep.

## Comfort and Control

- `acceleration(traj, dt)`: approximate per-step acceleration vectors.
- `jerk(traj, dt)`: approximate per-step jerk vectors.
- `acceleration_magnitude(traj, dt)`: per-step acceleration magnitudes.
- `jerk_magnitude(traj, dt)`: per-step jerk magnitudes.
- `jerk_cost(traj, dt)`: mean squared jerk magnitude.
- `max_acceleration(traj, dt)`: maximum acceleration magnitude.
- `mean_acceleration(traj, dt)`: mean acceleration magnitude.
- `rms_acceleration(traj, dt)`: root-mean-square acceleration magnitude.
- `max_deceleration(traj, dt)`: maximum longitudinal deceleration magnitude.
- `smoothness_score(traj)`: bounded shape score where `1.0` is smoother. The formula is `1 / (1 + log1p(cost))`, where `cost` is mean squared third finite difference normalized by mean squared step length. It is unitless, spatial-scale-invariant for geometrically similar paths, and separate from physical `jerk_cost(traj, dt)`. Trajectories shorter than four points return `1.0` with a `RuntimeWarning` because third finite differences are not measurable.

## Safety

- `collision_rate(ego_traj, actor_trajs, ego_radius, actor_radius)`: fraction of actor-covered ego timesteps colliding with any actor. For example, if the ego has five timesteps and the only actor has three, the denominator is the first three aligned ego timesteps.
- `time_to_collision(ego_state, actor_state, *, dt=None)`: constant-velocity disc-agent TTC. Use `AgentState`, dicts, flat `[x, y, vx, vy, radius]` arrays, or trajectory arrays with `dt`.
- `min_distance_to_actors(ego_traj, actor_trajs)`: minimum time-aligned XY distance to actors. Returns `math.inf` when no actor trajectories are provided.
- `lane_departure_rate(ego_traj, lane_boundary)`: fraction of ego points outside a polygonal lane boundary.
- `recovery_success_rate(opportunities, successes)`: successful recoveries divided by recovery opportunities. No opportunities returns `nan` because the denominator is undefined.
- `failure_severity(failures, aggregation="mean")`: mean or max severity over numeric non-negative severity values, or known category labels from `minor` through `fatal`. Empty failure collections return `0.0`.
- `near_miss_rate(clearances, threshold, collision_mask=None)`: fraction of clearance samples with `clearance < threshold` and no collision. Collisions are excluded from near-miss counts by default.
- `intervention_free_time(timestamps, interventions, mode="longest")`: longest or mean duration of consecutive non-intervention segments. Segment duration is `timestamp[last_false] - timestamp[first_false]`.

## Coverage

- `coverage_score(samples, bounds, bins=10)`: grid coverage for finite `NxD` samples. Bounds are `Dx2`, bins may be scalar or per-dimension, and the score is `occupied_bins / total_bins`. Duplicate samples do not increase coverage. Out-of-bounds samples are ignored.

## Calibration

- `calibration_error(confidences, correctness, n_bins=10)`: expected calibration error over uniform bins in `[0, 1]`. Lower is better. Confidence values must be probabilities and correctness labels must be boolean or 0/1.

## Physical Consistency

- `speed_profile(traj, dt)`: speed at each point. Returns length N for an N-point trajectory; endpoint values are finite-difference gradient estimates rather than `N-1` interval speeds.
- `acceleration_limits_violated(traj, dt, max_accel)`: thresholded acceleration result.
- `jerk_limits_violated(traj, dt, max_jerk)`: thresholded jerk result.
- `curvature_limits_violated(traj, max_curvature)`: thresholded curvature result.
- `dynamic_feasibility_score(traj, dt, constraints)`: `0..1` feasibility score for optional speed, acceleration, jerk, and curvature limits. The score is based on the worst relative violation, so adding satisfied constraints does not dilute an existing violation. Zero limits are accepted; positive observed motion against a zero limit scores `0.0`. Supported constraint keys are `max_speed`, `max_accel`, `max_jerk`, and `max_curvature`; unknown keys raise `ValueError`.
- `kinematic_feasibility(positions, dt=1.0, timestamps=None, max_velocity=None, max_acceleration=None, max_curvature=None)`: `0..1` score from finite-difference velocity, acceleration, and optional curvature limit checks. Higher is more feasible.
- `dynamic_feasibility(mass, accelerations, ...)`: `0..1` Newtonian feasibility score for optional max force, max acceleration, max torque, and friction-cone checks. Higher is more feasible and no configured constraints returns `1.0` after input validation.
- `physics_violation_rate(violations)`: fraction of timesteps/events with any violation. Mapping values are aggregated by logical OR; lower is better.

## Diversity

- `behavioral_diversity(behaviors, max_pairs=10000, normalize=False)`: mean pairwise Euclidean distance between unique behavior embeddings or flattened trajectories/actions. Duplicate behaviors do not inflate diversity. Pair sampling is deterministic when capped by `max_pairs`.

## Category Taxonomy

The registry categories are intentionally pragmatic. `comfort` contains
kinematic profiles and smoothness values used for ride/control quality,
`temporal` contains rollout drift and compounding-error metrics, and `physics`
contains dynamic limit checks such as `acceleration_limits_violated()` and
`dynamic_feasibility_score()`. Running all kinematic quantities may require
selecting multiple categories or explicit metric names.

## Edge-Case Behavior

- Empty arrays, one-dimensional arrays, invalid shapes, and trajectories with `NaN` or infinite values raise `ValueError`.
- Single-point trajectories are valid for metrics that can define a degenerate result, such as `path_length`, `curvature`, `speed_profile`, comfort metrics, and dynamic feasibility checks.
- `Nx3` trajectories are supported anywhere trajectory inputs accept `Nx2`; planar metrics such as `curvature` use the XY components.
- Thresholded physics metrics return structured `MetricResult` objects with `value`, `unit`, `passed`, `threshold`, and `metadata`.
- `min_distance_to_actors(ego_traj, [])` returns `math.inf` because there is no finite actor distance to report.
- `recovery_success_rate()` returns `nan` when there are no recovery opportunities.
- `failure_severity([])` returns `0.0`.
- `coverage_score()` ignores out-of-bounds samples instead of clipping them.
- `physics_violation_rate()` uses logical OR across violation types by default.

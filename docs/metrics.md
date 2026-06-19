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

## Physical Consistency

- `speed_profile(traj, dt)`: speed at each point. Returns length N for an N-point trajectory; endpoint values are finite-difference gradient estimates rather than `N-1` interval speeds.
- `acceleration_limits_violated(traj, dt, max_accel)`: thresholded acceleration result.
- `jerk_limits_violated(traj, dt, max_jerk)`: thresholded jerk result.
- `curvature_limits_violated(traj, max_curvature)`: thresholded curvature result.
- `dynamic_feasibility_score(traj, dt, constraints)`: `0..1` feasibility score for optional speed, acceleration, jerk, and curvature limits. The score is based on the worst relative violation, so adding satisfied constraints does not dilute an existing violation. Zero limits are accepted; positive observed motion against a zero limit scores `0.0`. Supported constraint keys are `max_speed`, `max_accel`, `max_jerk`, and `max_curvature`; unknown keys raise `ValueError`.

## Category Taxonomy

The registry categories are intentionally pragmatic. `comfort` contains
kinematic profiles and smoothness values used for ride/control quality, while
`physics` contains dynamic limit checks such as
`acceleration_limits_violated()` and `dynamic_feasibility_score()`. Running all
kinematic quantities may require selecting both categories or explicit metric
names.

## Edge-Case Behavior

- Empty arrays, one-dimensional arrays, invalid shapes, and trajectories with `NaN` or infinite values raise `ValueError`.
- Single-point trajectories are valid for metrics that can define a degenerate result, such as `path_length`, `curvature`, `speed_profile`, comfort metrics, and dynamic feasibility checks.
- `Nx3` trajectories are supported anywhere trajectory inputs accept `Nx2`; planar metrics such as `curvature` use the XY components.
- Thresholded physics metrics return structured `MetricResult` objects with `value`, `unit`, `passed`, `threshold`, and `metadata`.
- `min_distance_to_actors(ego_traj, [])` returns `math.inf` because there is no finite actor distance to report.

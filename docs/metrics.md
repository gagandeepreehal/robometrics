# Metrics

Built-in metrics are registered for evaluation through `robotmetrics.registry`. The evaluator uses short canonical names for common displacement metrics, including `ade` and `fde`, while aliases such as `average_displacement_error` and `final_displacement_error` remain available through registry lookup.

## Trajectory

- `average_displacement_error(pred, gt)`: mean pointwise Euclidean distance.
- `final_displacement_error(pred, gt)`: final-point Euclidean distance.
- `hausdorff_distance(pred, gt)`: symmetric Hausdorff distance.
- `path_length(traj)`: total Euclidean path length.
- `curvature(traj)`: approximate planar curvature at each point.
- `lateral_error(pred, ref)`: mean absolute perpendicular error from a reference path.
- `longitudinal_error(pred, ref)`: mean absolute along-track error from a reference path.

## Prediction

- `min_ade(predictions, gt)`: best average displacement error across modes.
- `min_fde(predictions, gt)`: best final displacement error across modes.
- `miss_rate(predictions, gt, threshold)`: returns `1.0` when all modes miss the final-point threshold, otherwise `0.0`.
- `topk_trajectory_error(predictions, gt, k)`: best ADE among the first `k` ranked modes.

## Comfort and Control

- `acceleration(traj, dt)`: approximate acceleration vectors.
- `jerk(traj, dt)`: approximate jerk vectors.
- `jerk_cost(traj, dt)`: mean squared jerk magnitude.
- `max_acceleration(traj, dt)`: maximum acceleration magnitude.
- `max_deceleration(traj, dt)`: maximum longitudinal deceleration magnitude.
- `smoothness_score(traj, dt)`: bounded score where `1.0` is smoother.

## Safety

- `collision_rate(ego_traj, actor_trajs, ego_radius, actor_radius)`: fraction of ego timesteps colliding with any actor.
- `time_to_collision(ego_state, actor_state)`: constant-velocity disc-agent TTC.
- `min_distance_to_actors(ego_traj, actor_trajs)`: minimum time-aligned XY distance to actors.
- `lane_departure_rate(ego_traj, lane_boundary)`: fraction of ego points outside a polygonal lane boundary.

## Physical Consistency

- `speed_profile(traj, dt)`: speed at each point.
- `acceleration_limits_violated(traj, dt, max_accel)`: thresholded acceleration result.
- `jerk_limits_violated(traj, dt, max_jerk)`: thresholded jerk result.
- `curvature_limits_violated(traj, max_curvature)`: thresholded curvature result.
- `dynamic_feasibility_score(traj, dt, constraints)`: `0..1` feasibility score for optional acceleration, jerk, and curvature limits.

# Safety Metrics

Use safety metrics for collision, separation, lane-boundary, drivable-area, and time-to-collision checks in navigation and driving-style evaluation.

## Reference
Hayward, Time-to-collision, 1972; Gottschalk et al., OBBTree, SIGGRAPH 1996; Caesar et al., nuScenes, CVPR 2020.

## Quick Example
```python
import numpy as np
from robometrics import AgentState, collision_rate, collision_rate_obb, min_distance_to_actors, time_to_collision

ego = np.array([[0.0, 0.0], [1.0, 0.0]])
actors = [np.array([[10.0, 0.0], [1.4, 0.0]])]
print(collision_rate(ego, actors, ego_radius=0.5, actor_radius=0.5))
print(min_distance_to_actors(ego, actors))

ego_state = AgentState(x=0.0, y=0.0, vx=2.0, vy=0.0, radius=0.5)
actor_state = AgentState(x=8.0, y=0.0, vx=0.0, vy=0.0, radius=0.5)
print(time_to_collision(ego_state, actor_state))

ego_dims = np.array([4.5, 2.0])
ego_yaws = np.array([0.0, 0.0])
actor_dims = [np.array([4.5, 2.0])]
actor_yaws = [np.array([0.0, 0.0])]
print(collision_rate_obb(ego, ego_dims, ego_yaws, actors, actor_dims, actor_yaws))
```

## Metrics

### `collision_rate(ego_traj, actor_trajs, ego_radius, actor_radius) -> float`

Formula: mean over actor-covered timesteps of any disc overlap.
Reference: Standard disc-overlap collision metric for robotics simulation
Unit: ratio
Direction: lower is better

Disc collision rate is a fast conservative check for round or buffered agents.

### `collision_rate_obb(ego_traj, ego_dims, ego_yaws, actor_trajs, actor_dims, actor_yaws) -> float`

Formula: mean over actor-covered timesteps of any SAT OBB overlap.
Reference: Gottschalk et al., OBBTree, SIGGRAPH 1996
Unit: ratio
Direction: lower is better

OBB collision rate handles oriented rectangular footprints for vehicles and mobile robots.

### `time_to_collision(ego_state, actor_state) -> float`

Formula: smallest nonnegative root of the relative-motion disc collision quadratic.
Reference: Hayward, Time-to-collision, 1972
Unit: seconds
Direction: higher is safer

Time to collision estimates imminent risk under constant velocity.
It accepts `AgentState`, dict, flat `[x, y, vx, vy, radius]` arrays, or
trajectory arrays when `dt` is supplied; trajectory inputs estimate velocity
from their first segment.

### `min_distance_to_actors(ego_traj, actor_trajs) -> float`

Formula: minimum time-aligned Euclidean distance from ego to any actor.
Reference: Standard minimum Euclidean separation metric
Unit: meters
Direction: higher is safer

Minimum actor distance reports the closest sampled encounter.

### `lane_departure_rate(ego_traj, lane_boundary) -> float`

Formula: mean of points outside a lane polygon.
Reference: Standard lane boundary containment metric
Unit: ratio
Direction: lower is better

Lane departure rate checks path containment in a provided lane boundary polygon.

### `offroad_rate(ego_traj, drivable_polygons) -> float`

Formula: mean of ego positions outside all drivable-area polygons.
Reference: Caesar et al., nuScenes, CVPR 2020
Unit: ratio
Direction: lower is better

Offroad rate evaluates whether sampled positions remain in drivable regions.

### `soft_ttc(ego_traj, actor_trajs, dt) -> float`

Formula: minimum constant-velocity TTC computed across rollout timesteps.
Reference: Weng et al., nuScenes-Forecast, ECCV 2022
Unit: seconds
Direction: higher is safer

Soft TTC summarizes the closest future interaction under local velocity estimates.

### `recovery_success_rate(opportunities, successes) -> float`

Formula: successful recoveries divided by recovery opportunities.
Reference: Standard recovery opportunity evaluation
Unit: ratio
Direction: higher is better

Recovery success rate reports how often a policy recovers when recovery was
available. No opportunities returns `nan`.

### `failure_severity(failures, aggregation="mean") -> float`

Formula: mean or max over numeric severity values or known severity labels.
Reference: Standard incident severity aggregation
Unit: severity
Direction: lower is better

Failure severity summarizes incident impact. Empty failure collections return
`0.0`.

### `near_miss_rate(clearances, threshold, collision_mask=None) -> float`

Formula: fraction of clearance samples below threshold, excluding collisions by default.
Reference: Standard near-miss clearance diagnostic
Unit: ratio
Direction: lower is better

Near-miss rate separates risky close calls from confirmed collisions.

### `intervention_free_time(timestamps, interventions, mode="longest") -> float`

Formula: longest or mean duration of consecutive non-intervention segments.
Reference: Standard autonomy intervention diagnostic
Unit: seconds
Direction: higher is better

Intervention-free time measures how long a system runs without human or safety
intervention.

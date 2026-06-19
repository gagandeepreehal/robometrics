# Safety Metrics
Use safety metrics for collision, separation, lane-boundary, drivable-area, and time-to-collision checks in navigation and driving-style evaluation.
## Reference
Hayward, Time-to-collision, 1972; Gottschalk et al., OBBTree, SIGGRAPH 1996; Caesar et al., nuScenes, CVPR 2020.
## Quick Example
```python
import numpy as np
from robometrics import collision_rate, min_distance_to_actors

ego = np.array([[0.0, 0.0], [1.0, 0.0]])
actors = [np.array([[10.0, 0.0], [1.4, 0.0]])]
print(collision_rate(ego, actors, ego_radius=0.5, actor_radius=0.5))
print(min_distance_to_actors(ego, actors))
```
Metrics
collision_rate(ego_traj, actor_trajs, ego_radius, actor_radius) -> float
Formula: mean over actor-covered timesteps of any disc overlap.
Reference: Standard disc-overlap collision metric for robotics simulation
Unit: ratio
Direction: lower is better

Disc collision rate is a fast conservative check for round or buffered agents.

collision_rate_obb(ego_traj, ego_dims, ego_yaws, actor_trajs, actor_dims, actor_yaws) -> float
Formula: mean over actor-covered timesteps of any SAT OBB overlap.
Reference: Gottschalk et al., OBBTree, SIGGRAPH 1996
Unit: ratio
Direction: lower is better

OBB collision rate handles oriented rectangular footprints for vehicles and mobile robots.

time_to_collision(ego_state, actor_state) -> float
Formula: smallest nonnegative root of the relative-motion disc collision quadratic.
Reference: Hayward, Time-to-collision, 1972
Unit: seconds
Direction: higher is safer

Time to collision estimates imminent risk under constant velocity.

min_distance_to_actors(ego_traj, actor_trajs) -> float
Formula: minimum time-aligned Euclidean distance from ego to any actor.
Reference: Standard minimum Euclidean separation metric
Unit: meters
Direction: higher is safer

Minimum actor distance reports the closest sampled encounter.

lane_departure_rate(ego_traj, lane_boundary) -> float
Formula: mean of points outside a lane polygon.
Reference: Standard lane boundary containment metric
Unit: ratio
Direction: lower is better

Lane departure rate checks path containment in a provided lane boundary polygon.

offroad_rate(ego_traj, drivable_polygons) -> float
Formula: mean of ego positions outside all drivable-area polygons.
Reference: Caesar et al., nuScenes, CVPR 2020
Unit: ratio
Direction: lower is better

Offroad rate evaluates whether sampled positions remain in drivable regions.

soft_ttc(ego_traj, actor_trajs, dt) -> float
Formula: minimum constant-velocity TTC computed across rollout timesteps.
Reference: Weng et al., nuScenes-Forecast, ECCV 2022
Unit: seconds
Direction: higher is safer

Soft TTC summarizes the closest future interaction under local velocity estimates.

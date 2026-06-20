# Physics Metrics
Use physics metrics to inspect sampled speed and to flag trajectories that violate acceleration, jerk, curvature, or combined dynamic feasibility constraints.
## Reference
Standard finite-difference kinematics and common robotics dynamic constraint checks.
## Quick Example
```python
import numpy as np
from robometrics import speed_profile, dynamic_feasibility_score

traj = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.1], [1.5, 0.2]])
dt = 0.1
print(speed_profile(traj, dt))
print(dynamic_feasibility_score(traj, dt, constraints={"max_speed": 8.0}))
```
Metrics
speed_profile(traj, dt) -> NDArray[np.float64]
Formula: norm of first finite-difference gradient divided by dt.
Reference: Standard finite-difference kinematics
Unit: m/s
Direction: context dependent

Speed profile exposes sampled velocity for limits and diagnostics. It returns
one speed estimate per trajectory point; endpoint values are gradient-estimated
rather than `N-1` interval speeds.

acceleration_limits_violated(traj, dt, max_accel) -> MetricResult
Formula: true when any acceleration magnitude exceeds max_accel.
Reference: Standard acceleration constraint violation metric
Unit: m/s^2
Direction: lower is better

Acceleration violation reports whether acceleration constraints are breached.

jerk_limits_violated(traj, dt, max_jerk) -> MetricResult
Formula: true when any jerk magnitude exceeds max_jerk.
Reference: Standard jerk constraint violation metric
Unit: m/s^3
Direction: lower is better

Jerk violation catches abrupt motion beyond configured limits.

curvature_limits_violated(traj, max_curvature) -> MetricResult
Formula: true when any curvature sample exceeds max_curvature.
Reference: Standard curvature constraint violation metric
Unit: 1/m
Direction: lower is better

Curvature violation identifies turns that exceed path constraints.

dynamic_feasibility_score(traj, dt) -> float
Formula: 1 minus normalized count of configured dynamic limit violations.
Reference: RoboMetrics internal heuristic
Unit: score
Direction: higher is better

Dynamic feasibility score combines several constraint checks into a scalar score.

kinematic_feasibility(positions, dt=1.0, timestamps=None, max_velocity=None, max_acceleration=None, max_curvature=None) -> float
Formula: fraction-style score over optional finite-difference velocity, acceleration, and curvature checks.
Reference: Standard kinematic feasibility checks
Unit: score
Direction: higher is better

Kinematic feasibility scores how often sampled motion remains within configured
kinematic limits.

dynamic_feasibility(mass, accelerations, ...) -> float
Formula: Newtonian feasibility score over optional force, acceleration, torque, and friction constraints.
Reference: Standard dynamics feasibility checks
Unit: score
Direction: higher is better

Dynamic feasibility checks force and friction-style physical constraints.

physics_violation_rate(violations) -> float
Formula: fraction of timesteps or events with any physics violation.
Reference: RoboMetrics internal violation aggregation
Unit: ratio
Direction: lower is better

Physics violation rate aggregates boolean or mapping-based violation signals.

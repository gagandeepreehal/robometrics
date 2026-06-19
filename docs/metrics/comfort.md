# Comfort Metrics
Use comfort metrics when evaluating passenger or payload smoothness, acceleration exposure, jerk exposure, and trajectory aggressiveness.
## Reference
Standard finite-difference kinematics and RMS or peak acceleration summaries used in robotics motion evaluation.
## Quick Example
```python
import numpy as np
from robometrics import acceleration, jerk

traj = np.array([[0.0, 0.0], [1.0, 0.0], [2.5, 0.0], [4.5, 0.0]])
dt = 0.1
print(acceleration(traj, dt))
print(jerk(traj, dt))
```
Metrics
acceleration(traj, dt) -> NDArray[np.float64]
Formula: second finite difference of position divided by dt squared.
Reference: Standard finite-difference kinematics
Unit: m/s^2
Direction: lower magnitude is smoother

Acceleration returns vector samples for downstream peak, mean, or RMS summaries.

jerk(traj, dt) -> NDArray[np.float64]
Formula: third finite difference of position divided by dt cubed.
Reference: Standard finite-difference kinematics
Unit: m/s^3
Direction: lower magnitude is smoother

Jerk exposes rapid acceleration changes that often correlate with discomfort.

jerk_cost(traj, dt) -> float
Formula: mean squared jerk magnitude.
Reference: RoboMetrics internal comfort cost
Unit: m^2/s^6
Direction: lower is better

Jerk cost is a scalar penalty for abrupt motion.

acceleration_magnitude(traj, dt) -> NDArray[np.float64]
Formula: norm of each acceleration vector.
Reference: Standard finite-difference kinematics
Unit: m/s^2
Direction: lower is smoother

Acceleration magnitude converts vector acceleration into scalar samples.

jerk_magnitude(traj, dt) -> NDArray[np.float64]
Formula: norm of each jerk vector.
Reference: Standard finite-difference kinematics
Unit: m/s^3
Direction: lower is smoother

Jerk magnitude provides per-step scalar jerk exposure.

max_acceleration(traj, dt) -> float
Formula: max acceleration magnitude.
Reference: Standard peak acceleration evaluation
Unit: m/s^2
Direction: lower is better

Max acceleration flags extreme spikes.

mean_acceleration(traj, dt) -> float
Formula: mean acceleration magnitude.
Reference: Standard mean acceleration evaluation
Unit: m/s^2
Direction: lower is better

Mean acceleration summarizes sustained aggressiveness.

rms_acceleration(traj, dt) -> float
Formula: root mean square acceleration magnitude.
Reference: Standard RMS acceleration evaluation
Unit: m/s^2
Direction: lower is better

RMS acceleration weights sustained and peak exposure.

max_deceleration(traj, dt) -> float
Formula: maximum negative longitudinal acceleration magnitude.
Reference: Standard peak deceleration evaluation
Unit: m/s^2
Direction: lower is better

Max deceleration highlights harsh braking.

smoothness_score(traj) -> float
Formula: scale-normalized score derived from third differences.
Reference: RoboMetrics internal heuristic
Unit: score
Direction: higher is better

Smoothness score maps trajectory regularity to a bounded score.

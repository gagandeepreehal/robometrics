# Task Metrics
Use task metrics for binary success, goal-reaching, and manipulation-style task completion signals that represent whether an autonomy system did what the scenario required.
## Reference
Anderson et al., Habitat, ICCV 2019; Mahler et al., Dex-Net 2.0, RSS 2017; ISO/TS 15066; Siciliano et al., Robotics, Springer 2009.
## Quick Example
```python
import numpy as np
from robometrics import task_success_rate, goal_reaching_accuracy

outcomes = np.array([1, 0, 1, 1])
positions = np.array([[0.0, 0.0], [1.0, 1.0]])
goals = np.array([[0.2, 0.0], [3.0, 3.0]])
print(task_success_rate(outcomes))
print(goal_reaching_accuracy(positions, goals, tolerance=0.5))
```
Metrics
task_success_rate(outcomes) -> float
Formula: mean of binary success indicators.
Reference: Standard binary task evaluation used in robotics benchmarks
Unit: ratio
Direction: higher is better

Task success rate reports aggregate completion across attempts.

goal_reaching_accuracy(positions, goals, tolerance) -> float
Formula: mean of norm(position - goal) <= tolerance.
Reference: Anderson et al., Habitat: A Platform for Embodied AI Research, ICCV 2019
Unit: ratio
Direction: higher is better

Goal-reaching accuracy measures whether final positions land within tolerance.

grasp_success_rate(attempts, successes) -> float
Formula: successes divided by attempts.
Reference: Mahler et al., Dex-Net 2.0, RSS 2017
Unit: ratio
Direction: higher is better

Grasp success rate reports manipulation completion reliability.

contact_richness(contact_forces) -> float
Formula: fraction of timesteps with force magnitude above threshold.
Reference: Handa et al., DexPilot, ICRA 2020
Unit: score
Direction: higher is better

Contact richness indicates whether meaningful contact occurred during manipulation.

force_limit_compliance(forces, max_force) -> float
Formula: fraction of force samples within max_force.
Reference: ISO/TS 15066 collaborative robot safety
Unit: score
Direction: higher is better

Force compliance checks whether interaction forces stay within limits.

joint_limit_violation_rate(joint_angles, lower_limits, upper_limits) -> float
Formula: fraction of configurations with any joint outside limits.
Reference: Siciliano et al., Robotics, Springer 2009
Unit: ratio
Direction: lower is better

Joint limit violation rate measures kinematic constraint failures.

end_effector_tracking_error(ee_traj, target_traj) -> float
Formula: mean Euclidean distance between end-effector and target path.
Reference: Siciliano et al., Robotics, Springer 2009
Unit: meters
Direction: lower is better

End-effector tracking error evaluates task-space path following.

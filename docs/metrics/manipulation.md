# Manipulation Metrics

Use manipulation metrics for gripper, arm, and contact-rich task evaluation where success, contact, force, joint limits, and end-effector tracking are first-class signals.

## Reference
Mahler et al., Dex-Net 2.0, RSS 2017; Handa et al., DexPilot, ICRA 2020; ISO/TS 15066; Siciliano et al., Robotics, Springer 2009.

## Quick Example
```python
import numpy as np
from robometrics import grasp_success_rate, end_effector_tracking_error

attempts = np.array([1, 1, 1])
successes = np.array([1, 0, 1])
ee = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
target = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
print(grasp_success_rate(attempts, successes))
print(end_effector_tracking_error(ee, target))
```

## Metrics

### `grasp_success_rate(attempts, successes) -> float`

Formula: successful grasps divided by grasp attempts.
Reference: Mahler et al., Dex-Net 2.0, RSS 2017
Unit: ratio
Direction: higher is better

Grasp success rate tracks grasp reliability across attempts.

### `contact_richness(contact_forces) -> float`

Formula: fraction of timesteps with contact force magnitude above threshold.
Reference: Handa et al., DexPilot, ICRA 2020
Unit: score
Direction: higher is better

Contact richness is useful for contact-rich manipulation and dexterous interaction.

### `force_limit_compliance(forces, max_force) -> float`

Formula: fraction of force samples at or below max_force.
Reference: ISO/TS 15066 collaborative robot safety
Unit: score
Direction: higher is better

Force compliance checks whether contact stayed within configured limits.

### `joint_limit_violation_rate(joint_angles, lower_limits, upper_limits) -> float`

Formula: fraction of configurations with any joint outside its allowed range.
Reference: Siciliano et al., Robotics, Springer 2009
Unit: ratio
Direction: lower is better

Joint limit violation rate flags infeasible robot configurations.

### `end_effector_tracking_error(ee_traj, target_traj) -> float`

Formula: mean Euclidean distance between end-effector samples and target samples.
Reference: Siciliano et al., Robotics, Springer 2009
Unit: meters
Direction: lower is better

End-effector tracking error measures task-space path following accuracy.

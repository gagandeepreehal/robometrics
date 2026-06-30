# Temporal Metrics

Use temporal metrics when error accumulation across rollout time matters more than a single final scalar.

## Reference
RoboMetrics internal heuristic for temporal error growth.

## Quick Example
```python
import numpy as np
from robometrics import compounding_error_index

errors = np.array([0.2, 0.25, 0.4, 0.8])
score = compounding_error_index(errors)
print(score)
```

## Metrics

### `temporal_drift(predicted, reference) -> float`

Formula: least-squares slope of per-timestep L2 error over timestep index.
Reference: RoboMetrics internal temporal drift diagnostic
Unit: error/timestep
Direction: lower is better

Temporal drift reports whether rollout error trends upward or downward over
time. Negative values mean error decreases over the rollout.

### `action_jerk(actions, dt) -> float`

Formula: mean squared L2 norm of the second finite difference of actions.
Reference: RoboMetrics internal control smoothness diagnostic
Unit: action^2/s^4
Direction: lower is better

Action jerk is a scalar control-sequence smoothness penalty.

### `control_smoothness(actions, dt) -> float`

Formula: 1 / (1 + action_jerk(actions, dt)).
Reference: RoboMetrics internal control smoothness score
Unit: score
Direction: higher is better

Control smoothness maps action jerk into a bounded score.

### `long_horizon_drift(predicted, reference) -> float`

Formula: later-weighted mean L2 rollout error with weights 1..T.
Reference: RoboMetrics internal long-horizon drift diagnostic
Unit: state units
Direction: lower is better

Long-horizon drift penalizes the same error more when it appears later in a
rollout.

### `compounding_error_index(errors) -> float`

Formula: final error divided by initial finite error, with stable handling near zero.
Reference: RoboMetrics internal heuristic
Unit: ratio
Direction: lower is better

Compounding error index highlights whether a rollout gets worse as prediction or control steps compound.

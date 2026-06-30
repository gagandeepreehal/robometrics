# Driving Metrics

Use driving metrics for road-agent forecasting and driving-specific safety checks, including ranked prediction modes, drivable-area containment, and soft time-to-collision.

## Reference
Chang et al., Argoverse, CVPR 2019; Caesar et al., nuScenes, CVPR 2020; Weng et al., nuScenes-Forecast, ECCV 2022.

## Quick Example
```python
import numpy as np
from scipy.special import logsumexp
from robometrics import displacement_at_k, offroad_rate, prediction_nll

predictions = np.array([[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [1.4, 0.0]]])
gt = np.array([[0.0, 0.0], [1.1, 0.0]])
polygon = np.array([[-1.0, -1.0], [2.0, -1.0], [2.0, 1.0], [-1.0, 1.0]])
log_weights = np.log(np.array([0.7, 0.3]))
log_weights = log_weights - logsumexp(log_weights)  # normalized log probabilities
print(displacement_at_k(predictions, gt, k=2))
print(prediction_nll(predictions, log_weights, gt))
print(offroad_rate(gt, [polygon]))
```

## Metrics

### `displacement_at_k(predictions, gt, k) -> float`

Formula: best displacement error among the first k ranked predictions.
Reference: Chang et al., Argoverse, CVPR 2019
Unit: meters
Direction: lower is better

Displacement-at-k evaluates ranked driving forecasts under benchmark submission limits.

### `prediction_nll(predictions, log_weights, gt) -> float`

Formula: negative log likelihood of the ground truth under a Gaussian mixture.
Reference: Thiede and Brahma, NeurIPS Workshop 2019
Unit: nats
Direction: lower is better

Prediction NLL captures confidence-weighted forecast quality.
`log_weights` must be a K-length array aligned with the K prediction modes.
Values are natural-log probabilities in nats; unnormalized log scores are also
accepted because the metric uses log-sum-exp internally.

### `offroad_rate(ego_traj, drivable_polygons) -> float`

Formula: fraction of ego positions outside all drivable polygons.
Reference: Caesar et al., nuScenes, CVPR 2020
Unit: ratio
Direction: lower is better

Offroad rate evaluates drivable-area compliance.

### `soft_ttc(ego_traj, actor_trajs, dt) -> float`

Formula: minimum constant-velocity time to collision across rollout timesteps.
Reference: Weng et al., nuScenes-Forecast, ECCV 2022
Unit: seconds
Direction: higher is better

Soft TTC gives a safety diagnostic for interacting traffic agents.

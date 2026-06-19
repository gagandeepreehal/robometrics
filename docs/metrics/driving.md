# Driving Metrics
Use driving metrics for road-agent forecasting and driving-specific safety checks, including ranked prediction modes, drivable-area containment, and soft time-to-collision.
## Reference
Chang et al., Argoverse, CVPR 2019; Caesar et al., nuScenes, CVPR 2020; Weng et al., nuScenes-Forecast, ECCV 2022.
## Quick Example
```python
import numpy as np
from robometrics import displacement_at_k, offroad_rate

predictions = np.array([[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [1.4, 0.0]]])
gt = np.array([[0.0, 0.0], [1.1, 0.0]])
polygon = np.array([[-1.0, -1.0], [2.0, -1.0], [2.0, 1.0], [-1.0, 1.0]])
print(displacement_at_k(predictions, gt, k=2))
print(offroad_rate(gt, [polygon]))
```
Metrics
displacement_at_k(predictions, gt, k) -> float
Formula: best displacement error among the first k ranked predictions.
Reference: Chang et al., Argoverse, CVPR 2019
Unit: meters
Direction: lower is better

Displacement-at-k evaluates ranked driving forecasts under benchmark submission limits.

prediction_nll(predictions, log_weights, gt) -> float
Formula: negative log likelihood of the ground truth under a Gaussian mixture.
Reference: Thiede and Brahma, NeurIPS Workshop 2019
Unit: nats
Direction: lower is better

Prediction NLL captures confidence-weighted forecast quality.

offroad_rate(ego_traj, drivable_polygons) -> float
Formula: fraction of ego positions outside all drivable polygons.
Reference: Caesar et al., nuScenes, CVPR 2020
Unit: ratio
Direction: lower is better

Offroad rate evaluates drivable-area compliance.

soft_ttc(ego_traj, actor_trajs, dt) -> float
Formula: minimum constant-velocity time to collision across rollout timesteps.
Reference: Weng et al., nuScenes-Forecast, ECCV 2022
Unit: seconds
Direction: higher is better

Soft TTC gives a safety diagnostic for interacting traffic agents.

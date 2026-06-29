# Trajectory Metrics
Use trajectory metrics when you need geometric error, distance, length, curvature, or Frenet-style tracking quality for one predicted path against one reference path.
## Reference
Alahi et al., Social Force, CVPR 2016; Hausdorff, Grundzuge der Mengenlehre, 1914; standard arc length and differential geometry formulas.
## Quick Example
```python
import numpy as np
from robometrics import average_displacement_error, final_displacement_error

pred = np.array([[0.0, 0.0], [1.2, 0.0], [2.0, 0.0]])
gt = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
ade = average_displacement_error(pred, gt)
fde = final_displacement_error(pred, gt)
print(ade, fde)
```
Metrics
ade(pred, gt) -> float
Formula: mean over t of norm(pred[t] - gt[t]).
Reference: Alahi et al., Social Force, CVPR 2016
Unit: meters
Direction: lower is better

Average displacement error summarizes pointwise tracking error over an entire path and is the default trajectory accuracy metric.

fde(pred, gt) -> float
Formula: norm(pred[-1] - gt[-1]).
Reference: Alahi et al., Social Force, CVPR 2016
Unit: meters
Direction: lower is better

Final displacement error focuses on endpoint quality and is useful for goal-directed forecasts.

hausdorff_distance(pred, gt) -> float
Formula: max of the two directed nearest-neighbor distances between point sets.
Reference: Hausdorff, Grundzuge der Mengenlehre, 1914
Unit: meters
Direction: lower is better

Hausdorff distance captures worst-case geometric mismatch between two sampled
paths. Unlike ADE and FDE, it treats trajectories as point sets: the two inputs
may have different sample counts as long as they use the same coordinate
dimensionality.

path_length(traj) -> float
Formula: sum over t of norm(traj[t + 1] - traj[t]).
Reference: Standard arc length formula for sampled trajectories
Unit: meters
Direction: context dependent

Path length reports total traveled distance and is usually interpreted against task or route expectations.

curvature(traj) -> NDArray[np.float64]
Formula: abs(x_prime * y_double_prime - y_prime * x_double_prime) / speed cubed.
Reference: Standard differential geometry curvature formula
Unit: 1/m
Direction: lower is smoother

Curvature returns a sampled profile for diagnosing sharp turns along a planar
path. Samples are treated as uniformly spaced; resample timestamped or
irregular trajectories before using curvature-based metrics.

mean_curvature(traj) -> float
Formula: mean finite curvature over trajectory samples.
Reference: Standard mean of differential geometry curvature over trajectory samples
Unit: 1/m
Direction: lower is smoother

Mean curvature provides a scalar smoothness proxy for turning behavior.

lateral_error(pred, ref) -> float
Formula: mean absolute deviation perpendicular to the local reference tangent.
Reference: Standard Frenet-frame lateral tracking error
Unit: meters
Direction: lower is better

Lateral error separates side-to-side path deviation from progress along the route.

longitudinal_error(pred, ref) -> float
Formula: mean absolute deviation projected onto the local reference tangent.
Reference: Standard Frenet-frame longitudinal tracking error
Unit: meters
Direction: lower is better

Longitudinal error measures along-track lead or lag relative to a reference path.

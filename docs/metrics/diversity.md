# Diversity Metrics
Use diversity metrics when multi-modal predictions should cover several plausible futures instead of collapsing to near-identical trajectories.
## Reference
Standard pairwise trajectory diversity metric.
## Quick Example
```python
import numpy as np
from robometrics import trajectory_diversity

predictions = np.array([[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]])
score = trajectory_diversity(predictions)
print(score)
```
Metrics
trajectory_diversity(predictions) -> float
Formula: mean pairwise ADE between predicted trajectory modes.
Reference: Standard pairwise trajectory diversity metric
Unit: meters
Direction: higher is better

Trajectory diversity measures spread across candidate futures, which is useful alongside accuracy metrics.

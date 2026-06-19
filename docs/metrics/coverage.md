# Coverage Metrics
Use coverage metrics when exploration, workspace visitation, or spatial diversity matters more than path-to-reference tracking error.
## Reference
Standard grid-cell workspace coverage metric.
## Quick Example
```python
import numpy as np
from robometrics import workspace_coverage

points = np.array([[0.0, 0.0], [0.2, 0.2], [1.2, 0.0], [2.0, 1.0]])
visited = workspace_coverage(points, cell_size=1.0)
print(visited)
```
Metrics
workspace_coverage(points) -> float
Formula: count of unique grid cells occupied by sampled positions.
Reference: Standard grid-cell workspace coverage metric
Unit: cells
Direction: higher is better

Workspace coverage counts how broadly a policy, robot, or dataset visits a discretized area.

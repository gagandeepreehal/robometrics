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
coverage_score(samples, bounds, bins) -> float
Formula: occupied finite grid bins divided by total bins inside bounds.
Reference: Standard bounded grid coverage metric
Unit: ratio
Direction: higher is better

Coverage score measures how much of a bounded sample space has been visited.
Duplicate samples do not increase coverage, and out-of-bounds samples are
ignored.

workspace_coverage(points) -> float
Formula: count of unique grid cells occupied by sampled positions.
Reference: Standard grid-cell workspace coverage metric
Unit: cells
Direction: higher is better

Workspace coverage counts how broadly a policy, robot, or dataset visits a discretized area.

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
Metrics
compounding_error_index(errors) -> float
Formula: final error divided by initial finite error, with stable handling near zero.
Reference: RoboMetrics internal heuristic
Unit: ratio
Direction: lower is better

Compounding error index highlights whether a rollout gets worse as prediction or control steps compound.

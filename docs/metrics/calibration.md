# Calibration Metrics
Use calibration metrics when a model emits confidence estimates and you need to know whether predicted probabilities match observed binary outcomes.
## Reference
Naeini et al., Obtaining Well Calibrated Probabilities Using Bayesian Binning, AAAI 2015.
## Quick Example
```python
import numpy as np
from robometrics import calibration_error

confidences = np.array([0.1, 0.8, 0.9, 0.4])
outcomes = np.array([0, 1, 1, 0])
ece = calibration_error(confidences, outcomes, n_bins=5)
print(ece)
```
Metrics
calibration_error(confidences, outcomes) -> float
Formula: sum over bins of bin_weight * abs(mean_confidence - empirical_accuracy).
Reference: Naeini et al., ECE, AAAI 2015
Unit: ratio
Direction: lower is better

Calibration error reports how far confidence estimates are from observed success frequencies.

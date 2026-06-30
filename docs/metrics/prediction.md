# Prediction Metrics

Use prediction metrics for multi-modal forecasting outputs where a model emits several candidate trajectories, confidence weights, or ranked modes for one future ground truth path.

## Reference
Gupta et al., Social GAN, CVPR 2018; Chang et al., Argoverse, CVPR 2019; Thiede and Brahma, 2019.

## Quick Example
```python
import numpy as np
from robometrics import min_ade, min_fde

predictions = np.array([[[0.0, 0.0], [2.0, 0.0]], [[0.0, 0.0], [1.1, 0.0]]])
gt = np.array([[0.0, 0.0], [1.0, 0.0]])
print(min_ade(predictions, gt))
print(min_fde(predictions, gt))
```

## Metrics

### `min_ade(predictions, gt) -> float`

Formula: min over modes of ADE(mode, gt).
Reference: Gupta et al., Social GAN, CVPR 2018
Unit: meters
Direction: lower is better

minADE evaluates whether any predicted mode closely follows the realized path.

### `min_fde(predictions, gt) -> float`

Formula: min over modes of FDE(mode, gt).
Reference: Gupta et al., Social GAN, CVPR 2018
Unit: meters
Direction: lower is better

minFDE measures best endpoint accuracy across candidate modes.

### `miss_rate(predictions, gt) -> float`

Formula: 1 if minFDE exceeds threshold, else 0.
Reference: Chang et al., Argoverse, CVPR 2019
Unit: ratio
Direction: lower is better

Miss rate converts endpoint error into a binary failure indicator.

### `topk_trajectory_error(predictions, gt, k) -> float`

Formula: best ADE among the first k ranked predictions.
Reference: Thiede and Brahma, Analyzing Failures of CVAE, 2019
Unit: meters
Direction: lower is better

Top-k trajectory error evaluates ranked candidate quality under a fixed budget.

### `prediction_nll(predictions, log_weights, gt) -> float`

Formula: negative log likelihood of gt under weighted Gaussian trajectory modes.
Reference: Thiede and Brahma, NeurIPS Workshop 2019
Unit: nats
Direction: lower is better

Prediction NLL evaluates probabilistic calibration and accuracy together.

### `displacement_at_k(predictions, gt, k) -> float`

Formula: ADE of the first k ranked predictions after selecting the best among them.
Reference: Chang et al., Argoverse, CVPR 2019
Unit: meters
Direction: lower is better

Displacement-at-k is useful when a benchmark allows only the first k submitted modes.

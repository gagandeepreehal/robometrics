from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robometrics import displacement_at_k, offroad_rate, prediction_nll, soft_ttc


def main() -> None:
    ground_truth = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    predictions = np.array(
        [
            [[0.0, 0.0], [1.2, 0.0], [2.4, 0.0]],
            [[0.0, 0.0], [1.0, 0.1], [2.0, 0.1]],
            [[0.0, 0.0], [0.8, 0.0], [1.7, 0.0]],
        ]
    )
    log_weights = np.log(np.array([0.4, 0.4, 0.2]))
    drivable_area = np.array([[-1.0, -1.0], [3.0, -1.0], [3.0, 1.0], [-1.0, 1.0]])
    actor = np.array([[8.0, 0.0], [9.0, 0.0], [10.0, 0.0]])

    print(f"Prediction NLL: {prediction_nll(predictions, log_weights, ground_truth):.3f}")
    print(f"Off-road rate: {offroad_rate(ground_truth, [drivable_area]):.3f}")
    print(f"Soft TTC: {soft_ttc(ground_truth, [actor], dt=1.0):.3f}")
    print(f"Displacement at K: {displacement_at_k(predictions, ground_truth, k=2):.3f}")


if __name__ == "__main__":
    main()

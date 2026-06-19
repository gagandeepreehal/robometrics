from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robometrics import ade, fde, min_ade, min_fde, miss_rate


def main() -> None:
    ground_truth = np.array([[0.0, 0.0], [1.0, 0.1], [2.0, 0.2]])
    prediction = np.array([[0.0, 0.0], [1.1, 0.0], [2.1, 0.1]])
    predictions = np.array(
        [
            prediction,
            [[0.0, 0.0], [1.0, 0.2], [2.0, 0.25]],
        ]
    )

    print(f"ADE: {ade(prediction, ground_truth):.3f} m")
    print(f"FDE: {fde(prediction, ground_truth):.3f} m")
    print(f"minADE: {min_ade(predictions, ground_truth):.3f} m")
    print(f"minFDE: {min_fde(predictions, ground_truth):.3f} m")
    print(f"Miss rate: {miss_rate(predictions, ground_truth, threshold=0.5):.3f}")


if __name__ == "__main__":
    main()

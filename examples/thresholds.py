from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robometrics import Evaluator


def main() -> None:
    pred = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    gt = np.array([[0.0, 0.0], [1.1, 0.0], [2.2, 0.0]])

    result = Evaluator().evaluate(
        prediction=pred,
        ground_truth=gt,
        metrics=["ade", "fde"],
        thresholds={"ade": 0.15, "fde": 0.15},
    )

    for metric in result.results:
        print(
            f"{metric.name}: value={metric.value:.3f} {metric.unit}, "
            f"threshold={metric.threshold}, passed={metric.passed}"
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robometrics import Evaluator


def main() -> None:
    evaluator = Evaluator()

    pred = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    gt = np.array([[0.0, 0.0], [1.1, 0.0], [2.2, 0.0]])
    trajectory_result = evaluator.evaluate(
        prediction=pred,
        ground_truth=gt,
        metrics=["ade", "fde"],
        thresholds={"ade": 0.5, "fde": 1.0},
    )

    predictions = np.array(
        [
            [[0.0, 0.0], [1.8, 0.0], [3.0, 0.0]],
            [[0.0, 0.0], [1.1, 0.0], [2.2, 0.0]],
        ]
    )
    prediction_result = evaluator.evaluate(
        prediction=predictions,
        ground_truth=gt,
        metrics=["min_ade", "min_fde", "miss_rate", "topk_trajectory_error"],
        threshold=0.5,
        thresholds={
            "min_ade": 0.5,
            "min_fde": 0.5,
            "miss_rate": 0.0,
            "topk_trajectory_error": 0.5,
        },
        metric_kwargs={"topk_trajectory_error": {"k": 2}},
    )

    print("Single-trajectory metrics")
    print(trajectory_result.to_markdown())
    print("\nMultimodal prediction metrics")
    print(prediction_result.to_markdown())


if __name__ == "__main__":
    main()

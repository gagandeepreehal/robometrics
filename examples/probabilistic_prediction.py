"""Evaluate a probabilistic multi-modal prediction."""

from __future__ import annotations

import numpy as np

from robometrics import Evaluator, prediction_nll


def main() -> None:
    ground_truth = np.array([[0.0, 0.0], [1.0, 0.1], [2.0, 0.0]])
    predictions = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.1]],
            [[0.0, 0.0], [1.8, 0.5], [3.0, 0.5]],
            [[0.0, 0.0], [0.6, -0.2], [1.1, -0.4]],
        ]
    )
    log_weights = np.log(np.array([0.65, 0.25, 0.10]))

    direct_nll = prediction_nll(predictions, log_weights, ground_truth)
    result = Evaluator().evaluate(
        prediction=predictions,
        ground_truth=ground_truth,
        log_weights=log_weights,
        metrics=["prediction_nll", "min_ade", "min_fde"],
        thresholds={"prediction_nll": 0.5, "min_ade": 0.5, "min_fde": 0.5},
    )

    print(f"Direct prediction NLL: {direct_nll:.3f}")
    print(result.to_markdown())


if __name__ == "__main__":
    main()

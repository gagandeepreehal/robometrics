"""Register and run a custom metric pack in an isolated registry."""

from __future__ import annotations

import numpy as np

from robometrics import Evaluator, MetricRegistry, load_pack


def final_lateral_error(prediction, ground_truth) -> float:
    pred = np.asarray(prediction, dtype=float)
    gt = np.asarray(ground_truth, dtype=float)
    if pred.ndim != 2 or gt.ndim != 2:
        raise ValueError("prediction and ground_truth must be trajectory arrays")
    return float(abs(pred[-1, 1] - gt[-1, 1]))


METRIC_PACK = [
    {
        "name": "final_lateral_error",
        "fn": final_lateral_error,
        "category": "trajectory",
        "unit": "meters",
        "required_inputs": ("prediction", "ground_truth"),
        "reference": "Example custom pack, 2026",
        "is_novel": True,
        "higher_is_better": False,
    }
]


def main() -> None:
    custom_registry = MetricRegistry()
    load_pack("__main__", registry=custom_registry)

    result = Evaluator(metric_registry=custom_registry).evaluate(
        prediction=np.array([[0.0, 0.0], [1.0, 0.3], [2.0, 0.4]]),
        ground_truth=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.1]]),
        metrics=["final_lateral_error"],
        thresholds={"final_lateral_error": 0.5},
    )

    print(result.to_markdown())


if __name__ == "__main__":
    main()

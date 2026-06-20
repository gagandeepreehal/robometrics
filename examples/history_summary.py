"""Summarize metric history across policy checkpoints."""

from __future__ import annotations

import numpy as np

from robometrics import EvaluationHistory, Evaluator


def main() -> None:
    evaluator = Evaluator()
    ground_truth = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    checkpoints = {
        0: np.array([[0.0, 0.0], [1.4, 0.0], [2.8, 0.0]]),
        50: np.array([[0.0, 0.0], [1.2, 0.0], [2.3, 0.0]]),
        100: np.array([[0.0, 0.0], [1.05, 0.0], [2.1, 0.0]]),
    }

    history = EvaluationHistory()
    for step, prediction in checkpoints.items():
        result = evaluator.evaluate(
            prediction=prediction,
            ground_truth=ground_truth,
            metrics=["ade", "fde"],
            thresholds={"ade": 0.5, "fde": 0.75},
        )
        history.record(step=step, result=result, label=f"checkpoint-{step}")

    print(history.to_markdown(metrics=["ade", "fde"]))
    print(f"Best ADE step: {history.best_step('ade')}")
    print(f"ADE trend: {history.trend('ade'):.6f}")


if __name__ == "__main__":
    main()

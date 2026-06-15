from __future__ import annotations

import numpy as np

from robotmetrics import Evaluator, average_displacement_error, collision_rate, jerk_cost


def main() -> None:
    pred = np.array([[0, 0], [1, 0], [2, 0]])
    gt = np.array([[0, 0], [1.1, 0], [2.1, 0]])

    ade = average_displacement_error(pred, gt)
    comfort = jerk_cost(pred, dt=0.1)
    collisions = collision_rate(pred, [], ego_radius=0.5, actor_radius=0.5)
    evaluation = Evaluator().evaluate(
        prediction=pred,
        ground_truth=gt,
        metrics=["ade", "fde"],
        thresholds={"ade": 1.0, "fde": 2.0},
    )

    print(f"ADE: {ade:.3f}")
    print(f"Jerk cost: {comfort:.3f}")
    print(f"Collision rate: {collisions:.3f}")
    print(evaluation.to_markdown())


if __name__ == "__main__":
    main()

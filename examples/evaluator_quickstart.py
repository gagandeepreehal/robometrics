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
    )

    print(result.summary())
    print(result.to_markdown())


if __name__ == "__main__":
    main()

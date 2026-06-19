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
        thresholds={"ade": 0.5, "fde": 1.0},
    )

    print("Dictionary:")
    print(result.to_dict())
    print("\nJSON:")
    print(result.to_json())
    print("\nMarkdown:")
    print(result.to_markdown())
    print("\nDataFrame:")
    try:
        print(result.to_dataframe())
    except ImportError as exc:
        print(exc)


if __name__ == "__main__":
    main()

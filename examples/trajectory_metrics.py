from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robometrics import curvature, path_length


def main() -> None:
    trajectory = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.1],
            [2.0, 0.4],
            [3.0, 0.9],
        ]
    )

    print(f"Path length: {path_length(trajectory):.3f} m")
    print(f"Mean curvature: {np.mean(curvature(trajectory)):.3f} 1/m")


if __name__ == "__main__":
    main()

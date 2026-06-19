from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robometrics import acceleration, jerk, jerk_cost


def main() -> None:
    trajectory = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [1.2, 0.1],
            [2.1, 0.3],
            [3.1, 0.6],
        ]
    )
    dt = 0.5

    print(f"Acceleration samples: {acceleration(trajectory, dt=dt)} m/s^2")
    print(f"Jerk samples: {jerk(trajectory, dt=dt)} m/s^3")
    print(f"Jerk cost: {jerk_cost(trajectory, dt=dt):.3f}")


if __name__ == "__main__":
    main()

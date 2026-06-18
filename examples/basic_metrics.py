from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robometrics import (
    average_displacement_error,
    collision_rate,
    final_displacement_error,
    jerk_cost,
    path_length,
    speed_profile,
)


def main() -> None:
    pred = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    gt = np.array([[0.0, 0.0], [1.1, 0.0], [2.2, 0.0]])
    actor = np.array([[5.0, 0.0], [5.0, 0.0], [5.0, 0.0]])

    print(f"ADE: {average_displacement_error(pred, gt):.3f} m")
    print(f"FDE: {final_displacement_error(pred, gt):.3f} m")
    print(f"Path length: {path_length(pred):.3f} m")
    print(f"Jerk cost: {jerk_cost(pred, dt=0.1):.3f}")
    print(f"Collision rate: {collision_rate(pred, [actor], 0.5, 0.5):.3f}")
    print(f"Speed profile: {speed_profile(pred, dt=0.1)} m/s")


if __name__ == "__main__":
    main()

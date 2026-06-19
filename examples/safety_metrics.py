from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robometrics import collision_rate, min_distance_to_actors


def main() -> None:
    ego = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    actors = [
        np.array([[0.0, 2.0], [1.0, 1.5], [2.0, 0.8], [3.0, 0.4]]),
        np.array([[0.0, -3.0], [1.0, -3.0], [2.0, -3.0], [3.0, -3.0]]),
    ]

    print(f"Minimum actor distance: {min_distance_to_actors(ego, actors):.3f} m")
    print(f"Collision rate: {collision_rate(ego, actors, ego_radius=0.3, actor_radius=0.3):.3f}")


if __name__ == "__main__":
    main()

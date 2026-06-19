from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robometrics import (
    contact_richness,
    end_effector_tracking_error,
    force_limit_compliance,
    grasp_success_rate,
    joint_limit_violation_rate,
)


def main() -> None:
    attempts = np.array([1, 1, 0, 1])
    successes = np.array([1, 0, 1, 1])
    contact_forces = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.5, 0.2, 0.0]])
    joint_angles = np.array([[0.0, 0.0], [0.2, -0.1], [1.2, 0.0]])
    lower_limits = np.array([-1.0, -1.0])
    upper_limits = np.array([1.0, 1.0])
    ee_traj = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.2], [1.0, 0.0, 0.0]])
    target_traj = np.array([[0.0, 0.0, 0.0], [0.5, 0.1, 0.2], [1.0, 0.0, 0.0]])

    print(f"Grasp success rate: {grasp_success_rate(attempts, successes):.3f}")
    print(f"Contact richness: {contact_richness(contact_forces, threshold=0.1):.3f}")
    print(f"Force compliance: {force_limit_compliance(contact_forces, max_force=1.0):.3f}")
    print(
        "Joint violation rate: "
        f"{joint_limit_violation_rate(joint_angles, lower_limits, upper_limits):.3f}"
    )
    print(f"EE tracking error: {end_effector_tracking_error(ee_traj, target_traj):.3f}")


if __name__ == "__main__":
    main()

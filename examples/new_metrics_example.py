from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robometrics import (
    action_jerk,
    behavioral_diversity,
    calibration_error,
    compounding_error_index,
    control_smoothness,
    coverage_score,
    dynamic_feasibility,
    failure_severity,
    intervention_free_time,
    kinematic_feasibility,
    long_horizon_drift,
    near_miss_rate,
    physics_violation_rate,
    recovery_success_rate,
    temporal_drift,
)


def main() -> None:
    reference = np.array([[0.0], [1.0], [2.0], [3.0]])
    predicted = np.array([[0.0], [1.1], [2.3], [3.6]])
    actions = np.array([[0.0], [0.0], [1.0], [0.0]])

    print(f"Temporal drift: {temporal_drift(predicted, reference):.3f}")
    print(f"Action jerk: {action_jerk(actions):.3f}")
    print(f"Control smoothness: {control_smoothness(actions):.3f}")
    print(f"Long-horizon drift: {long_horizon_drift(predicted, reference):.3f}")
    print(f"Compounding error index: {compounding_error_index(predicted, reference):.3f}")

    print(f"Recovery success rate: {recovery_success_rate([1, 1, 1], [1, 0, 1]):.3f}")
    print(f"Failure severity: {failure_severity([1.0, 2.0, 3.0]):.3f}")
    print(f"Near-miss rate: {near_miss_rate([0.2, 0.5, 1.5], threshold=1.0):.3f}")
    print(
        "Intervention-free time: "
        f"{intervention_free_time([0.0, 1.0, 2.0, 3.0], [0, 1, 0, 0]):.3f}"
    )

    samples = np.array([[0.1, 0.1], [0.8, 0.1], [0.8, 0.8]])
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    print(f"Coverage score: {coverage_score(samples, bounds=bounds, bins=2):.3f}")

    print(f"Calibration error: {calibration_error([0.25, 0.75], [0, 1], n_bins=2):.3f}")
    print(f"Kinematic feasibility: {kinematic_feasibility([0.0, 1.0, 2.0]):.3f}")
    print(
        "Dynamic feasibility: "
        f"{dynamic_feasibility(mass=2.0, accelerations=[[3.0]], max_force=10.0):.3f}"
    )
    print(f"Physics violation rate: {physics_violation_rate([[0, 1, 0], [0, 0, 1]]):.3f}")
    print(f"Behavioral diversity: {behavioral_diversity([[0.0, 0.0], [3.0, 4.0]]):.3f}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import numpy as np

from robometrics import (
    calibration_error,
    compounding_error_index,
    control_smoothness,
    kinematic_feasibility,
    long_horizon_drift,
    near_miss_rate,
    physics_violation_rate,
    temporal_drift,
)


def _rollout_quality_score(predicted: np.ndarray, reference: np.ndarray) -> float:
    penalty = (
        abs(temporal_drift(predicted, reference))
        + long_horizon_drift(predicted, reference)
        + compounding_error_index(predicted, reference)
    )
    return float(1.0 / (1.0 + penalty))


def test_perfect_rollout_outranks_noisy_and_constant_rollouts() -> None:
    reference = np.array([[0.0], [1.0], [2.0], [3.0]])
    perfect = reference.copy()
    noisy = reference + 0.1
    constant = np.zeros_like(reference)

    perfect_score = _rollout_quality_score(perfect, reference)
    noisy_score = _rollout_quality_score(noisy, reference)
    constant_score = _rollout_quality_score(constant, reference)

    assert perfect_score > noisy_score > constant_score


def test_all_component_improvements_raise_simple_control_score() -> None:
    smooth_actions = np.ones((5, 1), dtype=np.float64)
    rough_actions = np.array([[1.0], [-1.0], [1.0], [-1.0], [1.0]])

    assert control_smoothness(smooth_actions) > control_smoothness(rough_actions)
    assert kinematic_feasibility([0.0, 1.0, 2.0], max_velocity=2.0) > kinematic_feasibility(
        [0.0, 3.0, 6.0],
        max_velocity=2.0,
    )


def test_collision_or_physics_violations_do_not_score_better_than_safe_case() -> None:
    safe_clearances = np.array([2.0, 2.5, 3.0])
    risky_clearances = np.array([0.2, 0.5, 3.0])
    safe_physics = np.array([False, False, False])
    violating_physics = np.array([False, True, False])

    assert near_miss_rate(safe_clearances, threshold=1.0) < near_miss_rate(
        risky_clearances,
        threshold=1.0,
    )
    assert physics_violation_rate(safe_physics) < physics_violation_rate(violating_physics)


def test_new_metrics_are_deterministic_across_repeated_runs() -> None:
    confidences = np.array([0.25, 0.75])
    correctness = np.array([0, 1])
    predicted = np.array([[0.0], [1.0], [3.0]])
    reference = np.array([[0.0], [1.0], [2.0]])

    first = (
        calibration_error(confidences, correctness, n_bins=2),
        temporal_drift(predicted, reference),
        long_horizon_drift(predicted, reference),
    )
    second = (
        calibration_error(confidences, correctness, n_bins=2),
        temporal_drift(predicted, reference),
        long_horizon_drift(predicted, reference),
    )

    assert first == second

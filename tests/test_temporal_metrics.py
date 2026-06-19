from __future__ import annotations

import numpy as np
import pytest

from robometrics import (
    action_jerk,
    compounding_error_index,
    control_smoothness,
    long_horizon_drift,
    temporal_drift,
)


def _oracle_slope(errors: np.ndarray) -> float:
    time = np.arange(errors.shape[0], dtype=np.float64)
    centered_time = time - np.mean(time)
    return float(np.dot(errors - np.mean(errors), centered_time) / np.sum(centered_time**2))


def test_temporal_drift_perfect_and_hand_calculated_slope() -> None:
    reference = np.zeros((4, 1), dtype=np.float64)
    perfect = reference.copy()
    increasing = np.array([[0.0], [1.0], [2.0], [3.0]])
    decreasing = np.array([[3.0], [2.0], [1.0], [0.0]])

    assert temporal_drift(perfect, reference) == 0.0
    assert temporal_drift(increasing, reference) == pytest.approx(1.0)
    assert temporal_drift(decreasing, reference) == pytest.approx(-1.0)
    assert temporal_drift(increasing, reference) == pytest.approx(
        _oracle_slope(np.array([0.0, 1.0, 2.0, 3.0]))
    )


def test_temporal_drift_batch_average_and_edge_case() -> None:
    reference = np.zeros((2, 4, 1), dtype=np.float64)
    predicted = np.array(
        [
            [[0.0], [1.0], [2.0], [3.0]],
            [[0.0], [0.0], [0.0], [0.0]],
        ]
    )

    assert temporal_drift(predicted, reference) == pytest.approx(0.5)
    assert temporal_drift(np.array([[2.0]]), np.array([[0.0]])) == 0.0


def test_action_jerk_constant_linear_jump_and_dt_scaling() -> None:
    constant = np.ones((4, 1), dtype=np.float64)
    linear = np.array([[0.0], [1.0], [2.0], [3.0]])
    jump = np.array([[0.0], [0.0], [1.0], [0.0]])

    assert action_jerk(constant) == 0.0
    assert action_jerk(linear) == 0.0
    assert action_jerk(jump) == pytest.approx((1.0 + 4.0) / 2.0)
    assert action_jerk(jump, dt=0.5) == pytest.approx(action_jerk(jump) * 16.0)
    assert action_jerk(np.array([[0.0], [1.0]])) == 0.0


def test_control_smoothness_score_direction_and_exact_value() -> None:
    constant = np.ones((4, 1), dtype=np.float64)
    alternating = np.array([[1.0], [-1.0], [1.0], [-1.0]])

    assert control_smoothness(constant) == 1.0
    assert control_smoothness(alternating) == pytest.approx(1.0 / 17.0)
    assert control_smoothness(constant) > control_smoothness(alternating)


def test_long_horizon_drift_weights_later_errors() -> None:
    reference = np.zeros((3, 1), dtype=np.float64)
    early_error = np.array([[3.0], [0.0], [0.0]])
    late_error = np.array([[0.0], [0.0], [3.0]])

    assert long_horizon_drift(reference, reference) == 0.0
    assert long_horizon_drift(early_error, reference) == pytest.approx(3.0 / 6.0)
    assert long_horizon_drift(late_error, reference) == pytest.approx(9.0 / 6.0)
    assert long_horizon_drift(late_error, reference) > long_horizon_drift(
        early_error,
        reference,
    )


def test_compounding_error_index_exact_values_and_reference_input() -> None:
    constant = np.array([1.0, 1.0, 1.0, 1.0])
    growing = np.array([1.0, 2.0, 3.0, 4.0])
    reference = np.zeros((4, 1), dtype=np.float64)
    predicted = growing[:, None]

    assert compounding_error_index(np.zeros(4)) == 0.0
    assert compounding_error_index(constant) == 0.0
    assert compounding_error_index(growing) == pytest.approx(3.0 / 2.5)
    assert compounding_error_index(predicted, reference) == pytest.approx(3.0 / 2.5)
    assert compounding_error_index(growing) > compounding_error_index(constant)


def test_temporal_metrics_reject_invalid_inputs_and_are_deterministic() -> None:
    valid = np.zeros((3, 1), dtype=np.float64)

    with pytest.raises(ValueError, match="same shape"):
        temporal_drift(valid, np.zeros((4, 1)))
    with pytest.raises(ValueError, match="dt"):
        action_jerk(valid, dt=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        compounding_error_index(np.array([0.0, -1.0]))

    first = long_horizon_drift(valid + 1.0, valid)
    second = long_horizon_drift(valid + 1.0, valid)
    assert first == second

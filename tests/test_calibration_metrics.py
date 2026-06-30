from __future__ import annotations

import numpy as np
import pytest

from robometrics import calibration_error


def _oracle_ece(confidences: np.ndarray, correctness: np.ndarray, n_bins: int) -> float:
    total = float(confidences.size)
    error = 0.0
    for bin_index in range(n_bins):
        lower = bin_index / n_bins
        upper = (bin_index + 1) / n_bins
        if bin_index == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not np.any(mask):
            continue
        accuracy = float(np.mean(correctness[mask]))
        confidence = float(np.mean(confidences[mask]))
        error += float(np.count_nonzero(mask)) / total * abs(accuracy - confidence)
    return error


def test_calibration_error_perfect_and_hand_calculated_two_bin_case() -> None:
    perfect_confidences = np.array([0.0, 1.0])
    perfect_correctness = np.array([0, 1])
    confidences = np.array([0.25, 0.75])
    correctness = np.array([0, 1])

    assert calibration_error(perfect_confidences, perfect_correctness, n_bins=2) == 0.0
    assert calibration_error(confidences, correctness, n_bins=2) == pytest.approx(0.25)
    assert calibration_error(confidences, correctness, n_bins=2) == pytest.approx(
        _oracle_ece(confidences, correctness, n_bins=2)
    )


def test_calibration_error_prompt_examples_and_monotonicity() -> None:
    low_error_confidences = np.array([0.9, 0.8, 0.1, 0.2])
    calibrated_correctness = np.array([1, 1, 0, 0])
    overconfident_wrong = np.array([0.9, 0.8, 0.9, 0.8])
    wrong_correctness = np.array([0, 0, 0, 0])

    low_error = calibration_error(low_error_confidences, calibrated_correctness, n_bins=2)
    high_error = calibration_error(overconfident_wrong, wrong_correctness, n_bins=2)

    assert low_error == pytest.approx(0.15)
    assert high_error == pytest.approx(0.85)
    assert high_error > low_error


def test_calibration_error_batches_are_flattened_and_deterministic() -> None:
    confidences = np.array([[0.25, 0.75], [0.25, 0.75]])
    correctness = np.array([[0, 1], [0, 1]])

    assert calibration_error(confidences, correctness, n_bins=2) == pytest.approx(0.25)
    assert calibration_error(confidences, correctness, n_bins=2) == calibration_error(
        confidences,
        correctness,
        n_bins=2,
    )


def test_calibration_error_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="requires correctness or outcomes"):
        calibration_error([0.5])
    with pytest.raises(ValueError, match="provide either"):
        calibration_error([0.5], [1], outcomes=[1])
    with pytest.raises(ValueError, match="same shape"):
        calibration_error([0.5, 0.6], [1])
    with pytest.raises(ValueError, match="at least one value"):
        calibration_error([], [])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        calibration_error([1.2], [1])
    with pytest.raises(ValueError, match="0/1"):
        calibration_error([0.5], [2])
    with pytest.raises(ValueError, match="n_bins"):
        calibration_error([0.5], [1], n_bins=0)

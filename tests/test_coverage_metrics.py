from __future__ import annotations

import numpy as np
import pytest

from robometrics import coverage_score


def test_coverage_score_hand_calculated_bins_and_duplicates() -> None:
    samples = np.array(
        [
            [0.1, 0.1],
            [0.2, 0.2],
            [0.8, 0.1],
        ]
    )
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])

    assert coverage_score(samples, bounds=bounds, bins=[2, 2]) == pytest.approx(0.5)
    assert coverage_score(samples[:1], bounds=bounds, bins=[2, 2]) == pytest.approx(0.25)


def test_coverage_score_out_of_bounds_are_ignored() -> None:
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    out_of_bounds = np.array([[2.0, 2.0], [-1.0, 0.5]])
    mixed = np.array([[2.0, 2.0], [0.75, 0.75]])

    assert coverage_score(out_of_bounds, bounds=bounds, bins=2) == 0.0
    assert coverage_score(mixed, bounds=bounds, bins=2) == pytest.approx(0.25)


def test_coverage_score_upper_bound_maps_to_last_bin() -> None:
    samples = np.array([[1.0, 1.0]])
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])

    assert coverage_score(samples, bounds=bounds, bins=[2, 2]) == pytest.approx(0.25)


def test_coverage_score_monotonicity_and_determinism() -> None:
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    sparse = np.array([[0.1, 0.1]])
    broader = np.array([[0.1, 0.1], [0.8, 0.1], [0.8, 0.8]])

    assert coverage_score(broader, bounds=bounds, bins=2) > coverage_score(
        sparse,
        bounds=bounds,
        bins=2,
    )
    assert coverage_score(broader, bounds=bounds, bins=2) == coverage_score(
        broader,
        bounds=bounds,
        bins=2,
    )


def test_coverage_score_rejects_invalid_inputs() -> None:
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="NxD"):
        coverage_score(np.array([0.0, 1.0]), bounds=bounds)
    with pytest.raises(ValueError, match="Dx2"):
        coverage_score(np.array([[0.1, 0.1]]), bounds=np.array([[0.0, 1.0]]))
    with pytest.raises(ValueError, match="lower < upper"):
        coverage_score(np.array([[0.1, 0.1]]), bounds=np.array([[1.0, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="positive integer"):
        coverage_score(np.array([[0.1, 0.1]]), bounds=bounds, bins=[2, 0])

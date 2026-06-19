from __future__ import annotations

import numpy as np
import pytest

from robometrics import (
    ade,
    average_displacement_error,
    curvature,
    fde,
    final_displacement_error,
    hausdorff_distance,
    lateral_error,
    longitudinal_error,
    path_length,
)


def test_average_and_final_displacement_error() -> None:
    pred = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    gt = np.array([[0.0, 0.0], [1.1, 0.0], [2.1, 0.0]])

    assert average_displacement_error(pred, gt) == pytest.approx((0.0 + 0.1 + 0.1) / 3.0)
    assert final_displacement_error(pred, gt) == pytest.approx(0.1)
    assert ade(pred, gt) == average_displacement_error(pred, gt)
    assert fde(pred, gt) == final_displacement_error(pred, gt)


def test_displacement_errors_support_3d_trajectories() -> None:
    pred = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    gt = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]])

    assert average_displacement_error(pred, gt) == pytest.approx(0.5)
    assert final_displacement_error(pred, gt) == pytest.approx(1.0)


def test_hausdorff_supports_different_lengths() -> None:
    first = np.array([[0.0, 0.0], [1.0, 0.0]])
    second = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])

    assert hausdorff_distance(first, second) == pytest.approx(2.0)


def test_path_length_and_curvature_edge_cases() -> None:
    single = np.array([[1.0, 2.0]])
    stationary = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

    assert path_length(single) == 0.0
    assert path_length(np.array([[0.0, 0.0], [3.0, 4.0]])) == 5.0
    assert np.allclose(curvature(single), np.array([0.0]))
    assert np.allclose(curvature(stationary), np.zeros(3))


def test_lateral_and_longitudinal_error() -> None:
    ref = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    pred = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])

    assert lateral_error(pred, ref) == pytest.approx(1.0)
    assert longitudinal_error(pred, ref) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("pred", "gt"),
    [
        (np.empty((0, 2)), np.empty((0, 2))),
        (np.array([[0.0, np.nan]]), np.array([[0.0, 0.0]])),
        (np.array([[0.0, 0.0]]), np.array([[0.0, 0.0], [1.0, 1.0]])),
    ],
)
def test_displacement_errors_reject_invalid_inputs(pred: np.ndarray, gt: np.ndarray) -> None:
    with pytest.raises(ValueError):
        average_displacement_error(pred, gt)

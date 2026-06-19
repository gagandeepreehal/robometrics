from __future__ import annotations

from math import inf

import numpy as np
import pytest

from robometrics import (
    displacement_at_k,
    min_ade,
    offroad_rate,
    prediction_nll,
    soft_ttc,
)


def test_prediction_nll_single_mode_matching_ground_truth_is_near_zero() -> None:
    gt = np.array([[0.0, 0.0], [1.0, 0.0]])
    predictions = gt[None, :, :]

    assert prediction_nll(predictions, [0.0], gt) == pytest.approx(0.0)
    assert prediction_nll(predictions, [0.0], ground_truth=gt) == pytest.approx(0.0)


def test_offroad_rate_inside_and_outside_polygon() -> None:
    polygon = np.array([[-1.0, -1.0], [2.0, -1.0], [2.0, 1.0], [-1.0, 1.0]])
    inside = np.array([[0.0, 0.0], [1.0, 0.0]])
    outside = np.array([[3.0, 0.0], [4.0, 0.0]])

    assert offroad_rate(inside, [polygon]) == 0.0
    assert offroad_rate(outside, [polygon]) == 1.0
    assert offroad_rate(inside, []) == 1.0


def test_soft_ttc_stationary_collision_and_diverging_rollout() -> None:
    stationary = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    ego = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    actor = np.array([[10.0, 0.0], [11.0, 0.0], [12.0, 0.0]])

    assert soft_ttc(stationary, [stationary], dt=1.0) == 0.0
    assert soft_ttc(ego, [actor], dt=1.0) == inf


def test_displacement_at_k_matches_ranked_best_of_k_behavior() -> None:
    gt = np.array([[0.0, 0.0], [1.0, 0.0]])
    predictions = np.array(
        [
            [[0.0, 0.0], [2.0, 0.0]],
            [[0.0, 0.0], [1.2, 0.0]],
            [[0.0, 0.0], [1.0, 0.0]],
        ]
    )

    assert displacement_at_k(predictions, gt, k=1) == pytest.approx(0.5)
    assert displacement_at_k(predictions, gt, k=predictions.shape[0]) == pytest.approx(
        min_ade(predictions, gt)
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda: prediction_nll(np.empty((0, 2, 2)), [], np.empty((2, 2))),
        lambda: offroad_rate(np.empty((0, 2)), [np.array([[0.0, 0.0]])]),
        lambda: soft_ttc(np.empty((0, 2)), [np.empty((0, 2))], dt=1.0),
        lambda: displacement_at_k(np.empty((0, 2, 2)), np.empty((2, 2)), k=1),
    ],
)
def test_driving_metrics_reject_empty_inputs(call) -> None:
    with pytest.raises(ValueError):
        call()


def test_prediction_nll_rejects_shape_mismatches() -> None:
    gt = np.array([[0.0, 0.0], [1.0, 0.0]])
    predictions = gt[None, :, :]

    with pytest.raises(ValueError, match="log_weights length"):
        prediction_nll(predictions, [0.0, 0.0], gt)
    with pytest.raises(ValueError, match="timesteps/dimensions"):
        prediction_nll(predictions, [0.0], np.array([[0.0, 0.0]]))


def test_offroad_rate_rejects_invalid_polygon_shape() -> None:
    with pytest.raises(ValueError, match="at least 3 vertices"):
        offroad_rate([[0.0, 0.0]], [np.array([[0.0, 0.0], [1.0, 0.0]])])


def test_soft_ttc_rejects_non_xy_inputs() -> None:
    ego = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    actor = np.array([[10.0, 0.0], [11.0, 0.0], [12.0, 0.0]])

    with pytest.raises(ValueError, match="ego_traj must be an Nx2 array"):
        soft_ttc(ego, [actor], dt=1.0)


def test_displacement_at_k_rejects_invalid_k_and_shape_mismatch() -> None:
    gt = np.array([[0.0, 0.0], [1.0, 0.0]])
    predictions = gt[None, :, :]

    with pytest.raises(ValueError, match="k must be positive"):
        displacement_at_k(predictions, gt, k=0)
    with pytest.raises(ValueError, match="timesteps/dimensions"):
        displacement_at_k(predictions, np.array([[0.0, 0.0]]), k=1)

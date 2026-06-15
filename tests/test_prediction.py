from __future__ import annotations

import numpy as np
import pytest

from robotmetrics import min_ade, min_fde, miss_rate, topk_trajectory_error


def test_min_ade_and_min_fde_choose_best_mode() -> None:
    gt = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    predictions = np.array(
        [
            [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0], [2.1, 0.0]],
        ]
    )

    assert min_ade(predictions, gt) == pytest.approx(0.1 / 3.0)
    assert min_fde(predictions, gt) == pytest.approx(0.1)


def test_miss_rate_is_set_level_indicator() -> None:
    gt = np.array([[0.0, 0.0], [1.0, 0.0]])
    close = np.array([[[0.0, 0.0], [1.2, 0.0]], [[0.0, 0.0], [2.0, 0.0]]])
    far = np.array([[[0.0, 0.0], [3.0, 0.0]], [[0.0, 0.0], [2.0, 0.0]]])

    assert miss_rate(close, gt, threshold=0.5) == 0.0
    assert miss_rate(far, gt, threshold=0.5) == 1.0


def test_topk_trajectory_error_uses_ranked_prefix() -> None:
    gt = np.array([[0.0, 0.0], [1.0, 0.0]])
    predictions = np.array(
        [
            [[0.0, 0.0], [5.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0]],
        ]
    )

    assert topk_trajectory_error(predictions, gt, k=1) == pytest.approx(2.0)
    assert topk_trajectory_error(predictions, gt, k=2) == 0.0


@pytest.mark.parametrize(
    "predictions",
    [
        np.empty((0, 2, 2)),
        np.array([[[0.0, np.nan], [1.0, 0.0]]]),
        np.array([[[0.0, 0.0]]]),
    ],
)
def test_prediction_metrics_reject_invalid_inputs(predictions: np.ndarray) -> None:
    gt = np.array([[0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError):
        min_ade(predictions, gt)


def test_topk_rejects_invalid_k() -> None:
    gt = np.array([[0.0, 0.0], [1.0, 0.0]])
    predictions = np.array([[[0.0, 0.0], [1.0, 0.0]]])

    with pytest.raises(ValueError):
        topk_trajectory_error(predictions, gt, k=0)
    with pytest.raises(ValueError):
        topk_trajectory_error(predictions, gt, k=2)

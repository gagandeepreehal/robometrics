from __future__ import annotations

import numpy as np
import pytest

from robometrics import (
    calibration_error,
    compounding_error_index,
    registry,
    trajectory_diversity,
    workspace_coverage,
)


def test_registry_contains_core_metric_categories() -> None:
    required = {
        "trajectory",
        "prediction",
        "temporal",
        "comfort",
        "safety",
        "coverage",
        "calibration",
        "physics",
        "diversity",
        "task",
    }

    assert required.issubset(set(registry.categories()))
    assert all(registry.list_metrics(category=category) for category in required)


def test_compounding_error_index() -> None:
    assert compounding_error_index([1.0, 2.0, 4.0]) == 4.0
    assert compounding_error_index([0.0, 0.0]) == 0.0
    assert compounding_error_index([0.0, 1.0]) == float("inf")


def test_workspace_coverage_counts_occupied_cells() -> None:
    points = np.array([[0.1, 0.1], [0.2, 0.2], [1.2, 0.1]])

    assert workspace_coverage(points, cell_size=1.0) == 2.0


def test_calibration_error_perfect_and_miscalibrated() -> None:
    assert calibration_error([1.0, 0.0], [1.0, 0.0], n_bins=2) == 0.0
    assert calibration_error([1.0, 1.0], [0.0, 0.0], n_bins=2) == 1.0


def test_trajectory_diversity_pairwise_mean_ade() -> None:
    predictions = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [1.0, 1.0]],
        ]
    )

    assert trajectory_diversity(predictions) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "call",
    [
        lambda: compounding_error_index([]),
        lambda: workspace_coverage([]),
        lambda: calibration_error([], []),
        lambda: trajectory_diversity(np.empty((0, 2, 2))),
    ],
)
def test_core_category_metrics_reject_empty_inputs(call) -> None:
    with pytest.raises(ValueError):
        call()

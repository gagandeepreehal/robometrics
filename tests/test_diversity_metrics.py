from __future__ import annotations

from math import sqrt

import numpy as np
import pytest

from robometrics import behavioral_diversity


def test_behavioral_diversity_identical_trajectories_return_zero() -> None:
    trajectories = np.zeros((3, 4, 2), dtype=np.float64)

    assert behavioral_diversity(trajectories) == 0.0


def test_behavioral_diversity_hand_calculated_pairwise_distance() -> None:
    embeddings = np.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],
            [6.0, 8.0],
        ]
    )

    assert behavioral_diversity(embeddings) == pytest.approx((5.0 + 10.0 + 5.0) / 3.0)


def test_behavioral_diversity_duplicates_do_not_inflate_score() -> None:
    with_duplicate = np.array([[0.0], [3.0], [3.0]])
    without_duplicate = np.array([[0.0], [3.0]])

    assert behavioral_diversity(with_duplicate) == pytest.approx(3.0)
    assert behavioral_diversity(with_duplicate) == behavioral_diversity(without_duplicate)


def test_behavioral_diversity_normalization_and_monotonicity() -> None:
    close = np.array([[0.0, 0.0], [1.0, 0.0]])
    far = np.array([[0.0, 0.0], [3.0, 4.0]])

    assert behavioral_diversity(far, normalize=True) == pytest.approx(5.0 / sqrt(2.0))
    assert behavioral_diversity(far) > behavioral_diversity(close)


def test_behavioral_diversity_deterministic_pair_sampling() -> None:
    embeddings = np.arange(20, dtype=np.float64).reshape(10, 2)

    first = behavioral_diversity(embeddings, max_pairs=4)
    second = behavioral_diversity(embeddings, max_pairs=4)
    assert first == second


def test_behavioral_diversity_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="NxD or NxTxD"):
        behavioral_diversity(np.zeros((2, 2, 2, 2)))
    with pytest.raises(ValueError, match="at least one value"):
        behavioral_diversity(np.array([]))
    with pytest.raises(ValueError, match="max_pairs"):
        behavioral_diversity(np.array([[0.0], [1.0]]), max_pairs=0)

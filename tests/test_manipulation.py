from __future__ import annotations

import numpy as np
import pytest

from robometrics import (
    contact_richness,
    end_effector_tracking_error,
    force_limit_compliance,
    grasp_success_rate,
    joint_limit_violation_rate,
)


def test_grasp_success_rate_counts_attempted_successes() -> None:
    assert grasp_success_rate([1, 1, 1], [1, 0, 1]) == pytest.approx(2.0 / 3.0)
    assert np.isnan(grasp_success_rate([0, 0, 0], [1, 1, 1]))


def test_contact_richness_above_and_below_threshold() -> None:
    assert contact_richness([[1.0, 0.0], [0.0, 2.0]], threshold=0.5) == 1.0
    assert contact_richness([0.01, 0.02], threshold=0.1) == 0.0


def test_force_limit_compliance_within_and_exceeding_limit() -> None:
    assert force_limit_compliance([[1.0, 0.0], [0.0, 1.0]], max_force=2.0) == 1.0
    assert force_limit_compliance([3.0, 4.0], max_force=2.0) == 0.0


def test_joint_limit_violation_rate_midpoints_and_outside_limits() -> None:
    lower = np.array([-1.0, -2.0])
    upper = np.array([1.0, 2.0])
    midpoints = np.array([[0.0, 0.0], [0.5, -0.5]])
    outside = np.array([[-1.1, 0.0], [0.0, 2.1]])

    assert joint_limit_violation_rate(midpoints, lower, upper) == 0.0
    assert joint_limit_violation_rate(outside, lower, upper) == 1.0


def test_end_effector_tracking_error_identical_trajectories() -> None:
    traj = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0]])

    assert end_effector_tracking_error(traj, traj) == 0.0


@pytest.mark.parametrize(
    "call",
    [
        lambda: grasp_success_rate([], []),
        lambda: contact_richness([]),
        lambda: force_limit_compliance([], max_force=1.0),
        lambda: joint_limit_violation_rate(np.empty((0, 2)), [-1.0, -1.0], [1.0, 1.0]),
        lambda: end_effector_tracking_error(np.empty((0, 2)), np.empty((0, 2))),
    ],
)
def test_manipulation_metrics_reject_empty_inputs(call) -> None:
    with pytest.raises(ValueError):
        call()


def test_grasp_success_rate_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        grasp_success_rate([1, 0], [1])


def test_contact_and_force_metrics_reject_invalid_shapes_and_limits() -> None:
    with pytest.raises(ValueError, match="T-length or TxD"):
        contact_richness(np.zeros((1, 1, 1)))
    with pytest.raises(ValueError, match="threshold"):
        contact_richness([1.0], threshold=0.0)
    with pytest.raises(ValueError, match="max_force"):
        force_limit_compliance([1.0], max_force=0.0)


def test_joint_limit_violation_rate_rejects_shape_mismatch_and_bad_limits() -> None:
    with pytest.raises(ValueError, match="match joint_angles width"):
        joint_limit_violation_rate([[0.0, 0.0]], [-1.0], [1.0])
    with pytest.raises(ValueError, match="greater than lower_limits"):
        joint_limit_violation_rate([[0.0]], [1.0], [1.0])


def test_end_effector_tracking_error_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        end_effector_tracking_error([[0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]])

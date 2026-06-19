from __future__ import annotations

import numpy as np
import pytest

from robometrics import (
    failure_severity,
    intervention_free_time,
    near_miss_rate,
    recovery_success_rate,
)


def test_recovery_success_rate_exact_cases_and_empty_opportunities() -> None:
    opportunities = np.array([1, 1, 1, 1])
    successes = np.array([1, 0, 1, 1])

    assert recovery_success_rate(opportunities, successes) == pytest.approx(0.75)
    assert recovery_success_rate(opportunities, np.zeros(4)) == 0.0
    assert recovery_success_rate(opportunities, np.ones(4)) == 1.0
    assert np.isnan(recovery_success_rate(np.zeros(4), np.zeros(4)))
    assert np.isnan(recovery_success_rate(np.array([], dtype=bool), np.array([], dtype=bool)))


def test_recovery_success_rate_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="same shape"):
        recovery_success_rate(np.ones(3), np.ones(4))
    with pytest.raises(ValueError, match="0/1"):
        recovery_success_rate(np.array([1, 2]), np.array([1, 0]))


def test_failure_severity_numeric_category_and_monotonic_cases() -> None:
    assert failure_severity([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert failure_severity([1.0, 2.0, 3.0], aggregation="max") == pytest.approx(3.0)
    assert failure_severity([]) == 0.0
    assert failure_severity(["minor", "critical"]) == pytest.approx(2.5)
    assert failure_severity([3.0, 4.0]) > failure_severity([1.0, 2.0])


def test_failure_severity_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="aggregation"):
        failure_severity([1.0], aggregation="median")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        failure_severity([-1.0])
    with pytest.raises(ValueError, match="unknown failure category"):
        failure_severity(["unknown"])


def test_near_miss_rate_hand_calculated_collision_exclusion() -> None:
    clearances = np.array([0.2, 0.5, 1.5, 0.1])
    collisions = np.array([False, True, False, False])

    assert near_miss_rate(clearances, threshold=1.0, collision_mask=collisions) == 0.5
    assert near_miss_rate(clearances, threshold=0.3, collision_mask=collisions) == 0.5
    assert near_miss_rate(clearances, threshold=2.0, collision_mask=collisions) == 0.75


def test_near_miss_rate_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="threshold"):
        near_miss_rate([0.1], threshold=0.0)
    with pytest.raises(ValueError, match="same shape"):
        near_miss_rate([0.1, 0.2], threshold=1.0, collision_mask=[False])


def test_intervention_free_time_interval_convention_and_modes() -> None:
    timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    interventions = np.array([False, False, True, False, False, False])

    assert intervention_free_time(timestamps, interventions) == pytest.approx(2.0)
    assert intervention_free_time(timestamps, interventions, mode="mean") == pytest.approx(1.5)
    assert intervention_free_time(timestamps, np.ones(6, dtype=bool)) == 0.0
    assert intervention_free_time(timestamps, np.zeros(6, dtype=bool)) == pytest.approx(5.0)


def test_intervention_free_time_invalid_and_deterministic() -> None:
    timestamps = np.array([0.0, 1.0, 2.0])
    interventions = np.array([False, True, False])

    with pytest.raises(ValueError, match="strictly increasing"):
        intervention_free_time([0.0, 2.0, 1.0], interventions)
    with pytest.raises(ValueError, match="same shape"):
        intervention_free_time(timestamps, [False, True])
    with pytest.raises(ValueError, match="mode"):
        intervention_free_time(timestamps, interventions, mode="median")  # type: ignore[arg-type]

    assert intervention_free_time(timestamps, interventions) == intervention_free_time(
        timestamps,
        interventions,
    )

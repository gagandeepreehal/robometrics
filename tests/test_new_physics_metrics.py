from __future__ import annotations

import numpy as np
import pytest

from robometrics import dynamic_feasibility, kinematic_feasibility, physics_violation_rate


def test_kinematic_feasibility_velocity_cases() -> None:
    feasible = np.array([0.0, 1.0, 2.0])
    too_fast = np.array([0.0, 3.0, 6.0])

    assert kinematic_feasibility(feasible, dt=1.0, max_velocity=2.0) == 1.0
    assert kinematic_feasibility(too_fast, dt=1.0, max_velocity=2.0) == 0.0


def test_kinematic_feasibility_acceleration_and_exact_mixed_score() -> None:
    accelerating = np.array([0.0, 1.0, 4.0])

    assert kinematic_feasibility(accelerating, dt=1.0, max_acceleration=3.0) == 1.0
    assert kinematic_feasibility(accelerating, dt=1.0, max_acceleration=1.0) == 0.0
    assert kinematic_feasibility(
        accelerating,
        dt=1.0,
        max_velocity=2.0,
        max_acceleration=3.0,
    ) == pytest.approx(2.0 / 3.0)


def test_kinematic_feasibility_timestamps_curvature_and_invalid_inputs() -> None:
    positions = np.array([[0.0], [2.0], [4.0]])
    turn = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])

    assert kinematic_feasibility(
        positions,
        timestamps=[0.0, 2.0, 4.0],
        max_velocity=1.5,
    ) == 1.0
    assert kinematic_feasibility(turn, max_curvature=10.0) == 1.0

    with pytest.raises(ValueError, match="dt"):
        kinematic_feasibility(positions, dt=0.0, max_velocity=1.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        kinematic_feasibility(positions, timestamps=[0.0, 2.0, 1.0], max_velocity=1.0)
    with pytest.raises(ValueError, match="2 or 3 dimensions"):
        kinematic_feasibility(positions, max_curvature=1.0)


def test_kinematic_feasibility_no_measurable_checks_and_empty_inputs() -> None:
    assert kinematic_feasibility([0.0], max_velocity=1.0, max_acceleration=1.0) == 1.0

    with pytest.raises(ValueError, match="at least one value"):
        kinematic_feasibility([], max_velocity=1.0)


def test_dynamic_feasibility_force_and_acceleration_cases() -> None:
    accelerations = np.array([[3.0]])

    assert dynamic_feasibility(mass=2.0, accelerations=accelerations, max_force=10.0) == 1.0
    assert dynamic_feasibility(mass=2.0, accelerations=accelerations, max_force=5.0) == 0.0
    assert dynamic_feasibility(
        mass=2.0,
        accelerations=accelerations,
        max_acceleration=2.0,
    ) == 0.0


def test_dynamic_feasibility_torque_friction_and_invalid_inputs() -> None:
    accelerations = np.array([[0.0]])

    assert dynamic_feasibility(
        mass=1.0,
        accelerations=accelerations,
        torques=np.array([[2.0]]),
        max_torque=3.0,
    ) == 1.0
    assert dynamic_feasibility(
        mass=1.0,
        accelerations=accelerations,
        friction_coefficients=0.5,
        normal_forces=[10.0],
        tangential_forces=[4.0],
    ) == 1.0
    assert dynamic_feasibility(
        mass=1.0,
        accelerations=accelerations,
        friction_coefficients=0.5,
        normal_forces=[10.0],
        tangential_forces=[6.0],
    ) == 0.0

    with pytest.raises(ValueError, match="torques are required"):
        dynamic_feasibility(mass=1.0, accelerations=accelerations, max_torque=1.0)
    with pytest.raises(ValueError, match="provided together"):
        dynamic_feasibility(
            mass=1.0,
            accelerations=accelerations,
            friction_coefficients=0.5,
            normal_forces=[10.0],
        )


def test_dynamic_feasibility_explicit_forces_vector_friction_and_invalid_shapes() -> None:
    accelerations = np.array([[1.0, 0.0], [0.0, 1.0]])

    assert dynamic_feasibility(
        mass=2.0,
        accelerations=accelerations,
        forces=np.array([[1.0, 0.0], [0.0, 1.0]]),
        max_force=2.0,
    ) == 1.0
    assert dynamic_feasibility(
        mass=1.0,
        accelerations=accelerations,
        friction_coefficients=[0.5, 0.5],
        normal_forces=[10.0, 10.0],
        tangential_forces=np.array([[3.0, 4.0], [6.0, 0.0]]),
    ) == pytest.approx(0.5)

    with pytest.raises(ValueError, match="matching samples"):
        dynamic_feasibility(
            mass=1.0,
            accelerations=accelerations,
            friction_coefficients=0.5,
            normal_forces=[10.0],
            tangential_forces=np.array([[1.0, 0.0], [1.0, 0.0]]),
        )
    with pytest.raises(ValueError, match="non-negative"):
        dynamic_feasibility(
            mass=1.0,
            accelerations=accelerations,
            friction_coefficients=-0.1,
            normal_forces=[10.0, 10.0],
            tangential_forces=[1.0, 1.0],
        )
    with pytest.raises(ValueError, match="1D or 2D"):
        dynamic_feasibility(
            mass=1.0,
            accelerations=accelerations,
            friction_coefficients=0.5,
            normal_forces=[10.0, 10.0],
            tangential_forces=np.zeros((2, 1, 1)),
        )


def test_physics_violation_rate_hand_calculated_masks() -> None:
    violations = {
        "velocity": np.array([False, True, False, False]),
        "acceleration": np.array([False, False, True, False]),
    }

    assert physics_violation_rate(violations) == pytest.approx(0.5)
    assert physics_violation_rate(np.zeros(4, dtype=bool)) == 0.0
    assert physics_violation_rate(np.ones(4, dtype=bool)) == 1.0


def test_physics_violation_rate_invalid_and_deterministic() -> None:
    mask = np.array([[False, True, False], [False, False, True]])

    assert physics_violation_rate(mask) == pytest.approx(2.0 / 3.0)
    assert physics_violation_rate(mask) == physics_violation_rate(mask)

    with pytest.raises(ValueError, match="same shape"):
        physics_violation_rate({"a": [False, True], "b": [False]})
    with pytest.raises(ValueError, match="at least one mask"):
        physics_violation_rate({})

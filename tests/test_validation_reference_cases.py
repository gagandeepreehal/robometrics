from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from robometrics.registry import registry
from robometrics.results import MetricResult
from robometrics.schemas import AgentState


def test_reference_cases_cover_every_registered_metric() -> None:
    cases = _load_reference_cases()

    assert set(cases) == {metric.name for metric in registry.list_metrics()}
    for metric_name, case in cases.items():
        assert case["metric"] == metric_name
        assert case["derivation"]
        assert "expected" in case


def test_reference_cases_match_metric_outputs() -> None:
    cases = _load_reference_cases()
    inputs = _reference_inputs()

    for metric in registry.list_metrics():
        raw_value = metric.fn(**inputs[metric.name])
        if isinstance(raw_value, MetricResult):
            raw_value = raw_value.value

        expected = cases[metric.name]["expected"]
        if isinstance(expected, list):
            assert np.allclose(np.asarray(raw_value, dtype=np.float64), np.asarray(expected))
        else:
            assert float(raw_value) == pytest.approx(float(expected))


def _load_reference_cases() -> dict[str, dict[str, Any]]:
    root = Path(__file__).resolve().parents[1] / "validation"
    cases: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*/cases.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload:
            cases[item["metric"]] = item
    return cases


def _reference_inputs() -> dict[str, dict[str, Any]]:
    pred = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    gt = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    traj = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    curve = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]])
    predictions = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            [[0.0, 0.0], [1.0, 1.0], [2.0, 1.0]],
        ]
    )
    square = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    actor = np.array([[0.0, 2.0], [1.0, 0.4], [2.0, 2.0]])

    return {
        "ade": {"pred": pred, "gt": gt},
        "fde": {"pred": pred, "gt": gt},
        "hausdorff_distance": {"pred": pred, "gt": gt},
        "path_length": {"traj": traj},
        "curvature": {"traj": traj},
        "mean_curvature": {"traj": traj},
        "lateral_error": {"pred": gt, "ref": pred},
        "longitudinal_error": {"pred": gt, "ref": pred},
        "min_ade": {"predictions": predictions, "gt": gt},
        "min_fde": {"predictions": predictions, "gt": gt},
        "miss_rate": {"predictions": predictions, "gt": gt, "threshold": 0.5},
        "topk_trajectory_error": {"predictions": predictions, "gt": gt, "k": 2},
        "prediction_nll": {
            "predictions": predictions,
            "log_weights": np.log(np.array([0.5, 0.5])),
            "gt": gt,
        },
        "displacement_at_k": {"predictions": predictions, "ground_truth": gt, "k": 1},
        "temporal_drift": {
            "predicted": np.array([[[0.0], [1.0], [3.0]]]),
            "reference": np.array([[[0.0], [0.0], [0.0]]]),
        },
        "action_jerk": {"actions": np.array([[0.0], [1.0], [3.0]]), "dt": 1.0},
        "control_smoothness": {"actions": np.array([[0.0], [1.0], [3.0]]), "dt": 1.0},
        "long_horizon_drift": {
            "predicted": np.array([[[0.0], [1.0], [3.0]]]),
            "reference": np.array([[[0.0], [0.0], [0.0]]]),
        },
        "compounding_error_index": {"errors_or_predicted": np.array([0.0, 1.0, 3.0])},
        "acceleration": {"traj": curve, "dt": 1.0},
        "jerk": {"traj": curve, "dt": 1.0},
        "jerk_cost": {"traj": curve, "dt": 1.0},
        "acceleration_magnitude": {"traj": curve, "dt": 1.0},
        "jerk_magnitude": {"traj": curve, "dt": 1.0},
        "max_acceleration": {"traj": curve, "dt": 1.0},
        "mean_acceleration": {"traj": curve, "dt": 1.0},
        "rms_acceleration": {"traj": curve, "dt": 1.0},
        "max_deceleration": {"traj": curve, "dt": 1.0},
        "smoothness_score": {"traj": traj},
        "collision_rate": {
            "ego_traj": pred,
            "actor_trajs": [actor],
            "ego_radius": 0.5,
            "actor_radius": 0.5,
        },
        "collision_rate_obb": {
            "ego_traj": pred,
            "ego_dims": np.array([1.0, 1.0]),
            "ego_yaws": np.zeros(3),
            "actor_trajs": [actor],
            "actor_dims": [np.array([1.0, 1.0])],
            "actor_yaws": [np.zeros(3)],
        },
        "time_to_collision": {
            "ego_state": AgentState(x=0.0, y=0.0, vx=1.0, vy=0.0, radius=0.5),
            "actor_state": AgentState(x=3.0, y=0.0, vx=0.0, vy=0.0, radius=0.5),
        },
        "min_distance_to_actors": {"ego_traj": pred, "actor_trajs": [actor]},
        "lane_departure_rate": {
            "ego_traj": np.array([[0.0, 0.0], [2.0, 0.0]]),
            "lane_boundary": square,
        },
        "offroad_rate": {
            "ego_traj": np.array([[0.0, 0.0], [2.0, 0.0]]),
            "drivable_polygons": [square],
        },
        "soft_ttc": {
            "ego_traj": pred,
            "actor_trajs": [np.array([[3.0, 0.0], [3.0, 0.0], [3.0, 0.0]])],
            "dt": 1.0,
            "ego_radius": 0.5,
            "actor_radius": 0.5,
        },
        "recovery_success_rate": {
            "opportunities": np.array([1, 1, 0, 1]),
            "successes": np.array([1, 0, 1, 1]),
        },
        "failure_severity": {"failures": np.array([1.0, 3.0])},
        "near_miss_rate": {
            "clearances": np.array([0.2, 1.0, 0.3]),
            "threshold": 0.5,
            "collision_mask": np.array([0, 0, 1]),
        },
        "intervention_free_time": {
            "timestamps": np.array([0.0, 1.0, 2.0, 3.0]),
            "interventions": np.array([0, 0, 1, 0]),
        },
        "speed_profile": {"traj": traj, "dt": 1.0},
        "acceleration_limits_violated": {"traj": curve, "dt": 1.0, "max_accel": 2.0},
        "jerk_limits_violated": {"traj": curve, "dt": 1.0, "max_jerk": 4.0},
        "curvature_limits_violated": {"traj": traj, "max_curvature": 1.0},
        "dynamic_feasibility_score": {
            "traj": traj,
            "dt": 1.0,
            "constraints": {"max_speed": 2.0},
        },
        "kinematic_feasibility": {
            "positions": np.array([[0.0], [1.0], [3.0]]),
            "dt": 1.0,
            "max_velocity": 1.5,
        },
        "dynamic_feasibility": {
            "mass": 2.0,
            "accelerations": np.array([[1.0, 0.0], [3.0, 0.0]]),
            "max_force": 4.0,
        },
        "physics_violation_rate": {"violations": np.array([[0, 1, 0], [0, 0, 1]])},
        "coverage_score": {
            "samples": np.array([[0.25, 0.25], [0.75, 0.75]]),
            "bounds": np.array([[0.0, 1.0], [0.0, 1.0]]),
            "bins": 2,
        },
        "workspace_coverage": {
            "points": np.array([[0.0, 0.0], [1.2, 0.0], [1.9, 0.0]]),
            "cell_size": 1.0,
        },
        "calibration_error": {
            "confidences": np.array([0.25, 0.75]),
            "correctness": np.array([0, 1]),
            "n_bins": 2,
        },
        "behavioral_diversity": {"behaviors": np.array([[0.0, 0.0], [3.0, 4.0]])},
        "trajectory_diversity": {"predictions": predictions},
        "task_success_rate": {"outcomes": np.array([1, 0, 1])},
        "goal_reaching_accuracy": {
            "positions": np.array([[0.0, 0.0], [2.0, 0.0]]),
            "goals": np.array([[0.0, 0.0], [0.0, 0.0]]),
            "tolerance": 1.0,
        },
        "grasp_success_rate": {"attempts": np.array([1, 1, 0]), "successes": np.array([1, 0, 1])},
        "contact_richness": {
            "contact_forces": np.array([[0.0, 0.0], [1.0, 0.0]]),
            "threshold": 0.5,
        },
        "force_limit_compliance": {"forces": np.array([[1.0, 0.0], [3.0, 0.0]]), "max_force": 2.0},
        "joint_limit_violation_rate": {
            "joint_angles": np.array([[0.0, 0.0], [2.0, 0.0]]),
            "lower_limits": np.array([-1.0, -1.0]),
            "upper_limits": np.array([1.0, 1.0]),
        },
        "end_effector_tracking_error": {"ee_traj": pred, "target_traj": gt},
    }

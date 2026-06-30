#!/usr/bin/env python3
"""Emit deterministic RoboMetrics timing baselines as JSON."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import numpy as np

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robometrics import (  # noqa: E402
    Evaluator,
    __version__,
    collision_rate_obb,
    contact_richness,
    end_effector_tracking_error,
    force_limit_compliance,
    joint_limit_violation_rate,
    lane_departure_rate,
)

BASELINE_SCHEMA_VERSION = "1"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run small deterministic RoboMetrics timing cases and emit JSON.",
    )
    parser.add_argument(
        "--output",
        help="write JSON to this path instead of stdout",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="timed runs per case; default: 3",
    )
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

    payload = run_baselines(repeat=args.repeat)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


def run_baselines(*, repeat: int = 3) -> dict[str, Any]:
    """Run every built-in timing case and return a JSON-compatible payload."""
    cases = [
        _timed_case("geometry_safety", _geometry_safety_case, repeat=repeat),
        _timed_case("dataset_evaluator", _dataset_evaluator_case, repeat=repeat),
        _timed_case("manipulation_metrics", _manipulation_case, repeat=repeat),
    ]
    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "robometrics_version": __version__,
        "repeat": repeat,
        "unit": "seconds",
        "cases": cases,
    }


def _timed_case(
    name: str,
    fn: Callable[[], dict[str, Any]],
    *,
    repeat: int,
) -> dict[str, Any]:
    fn()
    runs = []
    metadata: dict[str, Any] = {}
    for _ in range(repeat):
        start = time.perf_counter()
        metadata = fn()
        runs.append(time.perf_counter() - start)

    return {
        "name": name,
        "runs": [float(value) for value in runs],
        "best": float(min(runs)),
        "mean": float(statistics.fmean(runs)),
        "metadata": metadata,
    }


def _geometry_safety_case() -> dict[str, Any]:
    n = 2_000
    ego = np.column_stack([np.linspace(0.0, 100.0, n), np.zeros(n)])
    lane = np.array([[-1.0, -5.0], [101.0, -5.0], [101.0, 5.0], [-1.0, 5.0]])
    actors = [ego + np.array([0.2, 0.0]) for _ in range(5)]
    yaws = np.zeros(n)
    dims = np.array([4.5, 2.0])

    lane_departure_rate(ego, lane)
    collision_rate_obb(
        ego_traj=ego,
        ego_dims=dims,
        ego_yaws=yaws,
        actor_trajs=actors,
        actor_dims=[dims for _ in actors],
        actor_yaws=[yaws for _ in actors],
    )
    return {
        "timesteps": n,
        "actors": len(actors),
    }


def _dataset_evaluator_case() -> dict[str, Any]:
    rng = np.random.default_rng(7)
    samples = 50
    timesteps = 20
    ground_truths = [
        np.column_stack([np.linspace(0.0, 20.0, timesteps), np.full(timesteps, lane)])
        for lane in np.linspace(-1.0, 1.0, samples)
    ]
    predictions = [
        trajectory + rng.normal(scale=0.15, size=trajectory.shape)
        for trajectory in ground_truths
    ]
    Evaluator().evaluate_dataset(
        predictions=predictions,
        ground_truths=ground_truths,
        metrics=["ade", "fde"],
        thresholds={"ade": 0.5, "fde": 1.0},
    )
    return {
        "samples": samples,
        "timesteps": timesteps,
        "metrics": ["ade", "fde"],
    }


def _manipulation_case() -> dict[str, Any]:
    rng = np.random.default_rng(11)
    episodes = 100
    timesteps = 30
    lower_limits = np.array([-1.0, -1.0, -1.0])
    upper_limits = np.array([1.0, 1.0, 1.0])

    for _ in range(episodes):
        target = rng.normal(scale=0.2, size=(timesteps, 3)).cumsum(axis=0)
        executed = target + rng.normal(scale=0.03, size=target.shape)
        forces = rng.normal(scale=0.4, size=(timesteps, 3))
        joints = rng.uniform(-1.1, 1.1, size=(timesteps, 3))

        end_effector_tracking_error(executed, target)
        force_limit_compliance(forces, max_force=1.5)
        joint_limit_violation_rate(joints, lower_limits, upper_limits)
        contact_richness(forces, threshold=0.1)
    return {
        "episodes": episodes,
        "timesteps": timesteps,
    }


if __name__ == "__main__":
    raise SystemExit(main())

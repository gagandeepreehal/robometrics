"""Small benchmark profile definitions for local CLI evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from robometrics._version import __version__
from robometrics.registry import MetricDefinition, registry
from robometrics.results import EvaluationResult, MetricResult


@dataclass(frozen=True)
class BenchmarkProfile:
    """Declarative benchmark profile metadata."""

    name: str
    metrics: tuple[str, ...]
    thresholds: Mapping[str, float]
    input_schema: str
    description: str
    limitations: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible benchmark profile payload."""
        return {
            "name": self.name,
            "metrics": list(self.metrics),
            "thresholds": dict(self.thresholds),
            "input_schema": self.input_schema,
            "description": self.description,
            "limitations": self.limitations,
        }


_PROFILES: dict[str, BenchmarkProfile] = {
    "trajectory_prediction_basic": BenchmarkProfile(
        name="trajectory_prediction_basic",
        metrics=("ade", "fde", "hausdorff_distance"),
        thresholds={"ade": 1.0, "fde": 2.0},
        input_schema="pred and gt are matching Nx2 or Nx3 trajectories in meters.",
        description="Basic trajectory prediction regression profile.",
        limitations="Does not evaluate multimodal uncertainty or scene semantics.",
    ),
    "mobile_robot_safety": BenchmarkProfile(
        name="mobile_robot_safety",
        metrics=("ade", "fde", "lateral_error", "longitudinal_error"),
        thresholds={"ade": 0.5, "fde": 1.0, "lateral_error": 0.5},
        input_schema="pred is the rollout and gt is the reference path, both Nx2 or Nx3.",
        description="Path-tracking proxy profile for mobile robot safety regressions.",
        limitations="Uses geometric tracking proxies; it does not infer actors or maps.",
    ),
    "manipulation_tracking": BenchmarkProfile(
        name="manipulation_tracking",
        metrics=("end_effector_tracking_error",),
        thresholds={"end_effector_tracking_error": 0.02},
        input_schema="pred is end-effector trajectory and gt is target trajectory.",
        description="End-effector tracking profile for manipulation policy checks.",
        limitations="Does not inspect contacts, joints, or task success without extra inputs.",
    ),
    "policy_regression_ci": BenchmarkProfile(
        name="policy_regression_ci",
        metrics=("ade", "fde"),
        thresholds={"ade": 0.25, "fde": 0.5},
        input_schema="pred and gt are matching Nx2 or Nx3 trajectories committed as CI fixtures.",
        description="Minimal deterministic profile suitable for CI regression gates.",
        limitations="Small fixtures catch regressions; they are not leaderboard evidence.",
    ),
}


class UnknownBenchmarkProfileError(KeyError):
    """Raised when a benchmark profile name is unknown."""


def list_profiles() -> list[BenchmarkProfile]:
    """Return all built-in benchmark profiles sorted by name."""
    return [_PROFILES[name] for name in sorted(_PROFILES)]


def get_profile(name: str) -> BenchmarkProfile:
    """Return one benchmark profile by name."""
    try:
        return _PROFILES[name]
    except KeyError as exc:
        raise UnknownBenchmarkProfileError(f"unknown benchmark profile: {name}") from exc


def run_profile(
    name: str,
    *,
    prediction: Any,
    ground_truth: Any,
    thresholds: Optional[Mapping[str, float]] = None,
) -> EvaluationResult:
    """Run a benchmark profile against loaded prediction and ground-truth arrays."""
    profile = get_profile(name)
    threshold_map = dict(profile.thresholds)
    threshold_map.update(thresholds or {})

    results = [
        _run_profile_metric(registry.get(metric_name), prediction, ground_truth, threshold_map)
        for metric_name in profile.metrics
    ]
    return EvaluationResult(
        results=results,
        metadata={
            "robometrics_version": __version__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "benchmark_profile": profile.to_dict(),
        },
    )


def _run_profile_metric(
    metric: MetricDefinition,
    prediction: Any,
    ground_truth: Any,
    thresholds: Mapping[str, float],
) -> MetricResult:
    kwargs = _profile_metric_kwargs(metric, prediction, ground_truth)
    try:
        raw = metric.fn(**kwargs)
        value = raw.value if isinstance(raw, MetricResult) else _coerce_scalar(raw)
        metadata = dict(raw.metadata) if isinstance(raw, MetricResult) else {}
        metadata.update(
            {
                "category": metric.category,
                "description": metric.description,
                "reference": metric.reference,
                "higher_is_better": metric.higher_is_better,
            }
        )
        result = MetricResult(name=metric.name, value=value, unit=metric.unit, metadata=metadata)
    except Exception as exc:  # noqa: BLE001 - profile output should capture per-metric failures.
        result = MetricResult(
            name=metric.name,
            value=float("nan"),
            unit=metric.unit,
            metadata={
                "category": metric.category,
                "description": metric.description,
                "reference": metric.reference,
                "higher_is_better": metric.higher_is_better,
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )

    threshold = thresholds.get(metric.name)
    if threshold is not None and "error" not in result.metadata:
        result.threshold = float(threshold)
        result.passed = (
            bool(result.value >= result.threshold)
            if metric.higher_is_better
            else bool(result.value <= result.threshold)
        )
    return result


def _profile_metric_kwargs(
    metric: MetricDefinition,
    prediction: Any,
    ground_truth: Any,
) -> dict[str, Any]:
    values = {
        "pred": prediction,
        "prediction": prediction,
        "traj": prediction,
        "ego_traj": prediction,
        "ee_traj": prediction,
        "gt": ground_truth,
        "ground_truth": ground_truth,
        "ref": ground_truth,
        "reference": ground_truth,
        "target_traj": ground_truth,
    }
    missing = [name for name in metric.required_inputs if name not in values]
    if missing:
        raise ValueError(
            f"profile metric {metric.name!r} requires unsupported inputs: {', '.join(missing)}"
        )
    kwargs = {name: values[name] for name in metric.required_inputs}
    kwargs.update(metric.default_kwargs)
    return kwargs


def _coerce_scalar(value: Any) -> float:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return float(arr)
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite)) if finite.size else float("nan")

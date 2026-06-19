"""Metric registry for RoboMetrics."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Optional

import numpy as np

from robometrics import (
    calibration,
    comfort,
    coverage,
    diversity,
    driving,
    manipulation,
    physics,
    prediction,
    safety,
    task,
    temporal,
    trajectory,
)

MetricFn = Callable[..., Any]
CompatibilityFn = Callable[[Mapping[str, Any]], bool]


class UnknownMetricError(KeyError):
    """Raised when a metric name is not registered."""


@dataclass(frozen=True)
class MetricDefinition:
    """Registered metric metadata and execution contract."""

    name: str
    fn: MetricFn
    category: str
    unit: str
    description: str = ""
    required_inputs: tuple[str, ...] = ()
    default_kwargs: Mapping[str, Any] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    reference: str = ""
    is_novel: bool = False
    higher_is_better: bool = False
    compatibility: Optional[CompatibilityFn] = field(default=None, repr=False, compare=False)

    def is_compatible(self, inputs: Mapping[str, Any]) -> bool:
        """Return True when the available inputs can run this metric."""
        if any(key not in inputs or inputs[key] is None for key in self.required_inputs):
            return False
        if self.compatibility is None:
            return True
        return self.compatibility(inputs)


class MetricRegistry:
    """Registry for named metric callables."""

    def __init__(self) -> None:
        self._metrics: dict[str, MetricDefinition] = {}
        self._aliases: dict[str, str] = {}
        self._lock = RLock()

    def register(
        self,
        *,
        name: str,
        fn: MetricFn,
        category: str,
        unit: str = "",
        description: Optional[str] = None,
        required_inputs: Iterable[str] = (),
        default_kwargs: Optional[Mapping[str, Any]] = None,
        aliases: Iterable[str] = (),
        reference: str = "",
        is_novel: bool = False,
        higher_is_better: Optional[bool] = None,
        compatibility: Optional[CompatibilityFn] = None,
    ) -> MetricDefinition:
        """Register a metric function and return its definition."""
        with self._lock:
            normalized_name = _normalize_name(name)
            if normalized_name in self._metrics or normalized_name in self._aliases:
                raise ValueError(f"metric already registered: {name}")

            metric = MetricDefinition(
                name=normalized_name,
                fn=fn,
                category=category,
                unit=unit,
                description=description or _first_doc_line(fn),
                required_inputs=tuple(required_inputs),
                default_kwargs=dict(default_kwargs or {}),
                aliases=tuple(aliases),
                reference=reference,
                is_novel=bool(is_novel),
                higher_is_better=(
                    _default_higher_is_better(normalized_name)
                    if higher_is_better is None
                    else bool(higher_is_better)
                ),
                compatibility=compatibility,
            )
            self._metrics[normalized_name] = metric

            for alias in aliases:
                normalized_alias = _normalize_name(alias)
                if normalized_alias in self._metrics or normalized_alias in self._aliases:
                    raise ValueError(f"metric alias already registered: {alias}")
                self._aliases[normalized_alias] = normalized_name

            return metric

    def register_many(
        self,
        definitions: list[dict[str, Any]],
    ) -> list[MetricDefinition]:
        """Register multiple metrics from a list of definition dicts.

        Each dict must contain at minimum: name, fn, category.
        Optional keys: unit, description, required_inputs, default_kwargs,
        aliases, compatibility, reference, is_novel, higher_is_better.
        Returns a list of created MetricDefinition objects.
        """
        created = []
        required = ("name", "fn", "category")
        for definition in definitions:
            for key in required:
                if key not in definition:
                    raise ValueError(f"metric definition is missing required key: {key}")
            created.append(
                self.register(
                    name=definition["name"],
                    fn=definition["fn"],
                    category=definition["category"],
                    unit=definition.get("unit", ""),
                    description=definition.get("description"),
                    required_inputs=definition.get("required_inputs", ()),
                    default_kwargs=definition.get("default_kwargs"),
                    aliases=definition.get("aliases", ()),
                    compatibility=definition.get("compatibility"),
                    reference=definition.get("reference", ""),
                    is_novel=definition.get("is_novel", False),
                    higher_is_better=definition.get("higher_is_better"),
                )
            )
        return created

    def unregister(self, name: str) -> None:
        """Remove a registered metric and all its aliases.

        Raises UnknownMetricError if the metric does not exist.
        Used primarily for testing and pack development.
        """
        with self._lock:
            normalized_name = _normalize_name(name)
            canonical = self._aliases.get(normalized_name, normalized_name)
            if canonical not in self._metrics:
                raise UnknownMetricError(f"unknown metric: {name}")

            del self._metrics[canonical]
            for alias, target in list(self._aliases.items()):
                if target == canonical:
                    del self._aliases[alias]

    def get(self, name: str) -> MetricDefinition:
        """Return a metric definition by canonical name or alias."""
        with self._lock:
            normalized_name = _normalize_name(name)
            canonical = self._aliases.get(normalized_name, normalized_name)
            try:
                return self._metrics[canonical]
            except KeyError as exc:
                raise UnknownMetricError(f"unknown metric: {name}") from exc

    def list_metrics(self, *, category: Optional[str] = None) -> list[MetricDefinition]:
        """Return registered metrics, optionally filtered by category."""
        with self._lock:
            if category is None:
                return list(self._metrics.values())
            normalized_category = category.lower()
            return [
                metric
                for metric in self._metrics.values()
                if metric.category.lower() == normalized_category
            ]

    def categories(self) -> list[str]:
        """Return registered metric categories."""
        with self._lock:
            return sorted({metric.category for metric in self._metrics.values()})


def create_default_registry() -> MetricRegistry:
    """Create the default registry for all built-in metric functions."""
    reg = MetricRegistry()

    reg.register(
        name="ade",
        aliases=("average_displacement_error",),
        fn=trajectory.average_displacement_error,
        category="trajectory",
        unit="meters",
        required_inputs=("pred", "gt"),
        reference="Alahi et al., Social Force, CVPR 2016",
        compatibility=_same_shape("pred", "gt"),
    )
    reg.register(
        name="fde",
        aliases=("final_displacement_error",),
        fn=trajectory.final_displacement_error,
        category="trajectory",
        unit="meters",
        required_inputs=("pred", "gt"),
        reference="Alahi et al., Social Force, CVPR 2016",
        compatibility=_same_shape("pred", "gt"),
    )
    reg.register(
        name="hausdorff_distance",
        fn=trajectory.hausdorff_distance,
        category="trajectory",
        unit="meters",
        required_inputs=("pred", "gt"),
        reference="Hausdorff, Grundzüge der Mengenlehre, 1914",
        compatibility=_same_dimensionality("pred", "gt"),
    )
    reg.register(
        name="path_length",
        fn=trajectory.path_length,
        category="trajectory",
        unit="meters",
        required_inputs=("traj",),
        reference="Standard arc length formula for sampled trajectories",
        compatibility=_trajectory_input("traj"),
    )
    reg.register(
        name="curvature",
        fn=trajectory.curvature,
        category="trajectory",
        unit="1/m",
        required_inputs=("traj",),
        reference="Standard differential geometry curvature formula",
        compatibility=_trajectory_input("traj"),
    )
    reg.register(
        name="mean_curvature",
        fn=trajectory.mean_curvature,
        category="trajectory",
        unit="1/m",
        required_inputs=("traj",),
        reference="Standard mean of differential geometry curvature over trajectory samples",
        compatibility=_trajectory_input("traj"),
    )
    reg.register(
        name="lateral_error",
        fn=trajectory.lateral_error,
        category="trajectory",
        unit="meters",
        required_inputs=("pred", "ref"),
        reference="Standard Frenet-frame lateral tracking error",
        compatibility=_same_shape("pred", "ref"),
    )
    reg.register(
        name="longitudinal_error",
        fn=trajectory.longitudinal_error,
        category="trajectory",
        unit="meters",
        required_inputs=("pred", "ref"),
        reference="Standard Frenet-frame longitudinal tracking error",
        compatibility=_same_shape("pred", "ref"),
    )

    reg.register(
        name="min_ade",
        aliases=("minade",),
        fn=prediction.min_ade,
        category="prediction",
        unit="meters",
        required_inputs=("predictions", "gt"),
        reference="Gupta et al., Social GAN, CVPR 2018",
        compatibility=_prediction_matches_ground_truth("predictions", "gt"),
    )
    reg.register(
        name="min_fde",
        aliases=("minfde",),
        fn=prediction.min_fde,
        category="prediction",
        unit="meters",
        required_inputs=("predictions", "gt"),
        reference="Gupta et al., Social GAN, CVPR 2018",
        compatibility=_prediction_matches_ground_truth("predictions", "gt"),
    )
    reg.register(
        name="miss_rate",
        fn=prediction.miss_rate,
        category="prediction",
        unit="ratio",
        required_inputs=("predictions", "gt"),
        default_kwargs={"threshold": 2.0},
        reference="Chang et al., Argoverse, CVPR 2019",
        compatibility=_prediction_matches_ground_truth("predictions", "gt"),
    )
    reg.register(
        name="topk_trajectory_error",
        fn=prediction.topk_trajectory_error,
        category="prediction",
        unit="meters",
        required_inputs=("predictions", "gt"),
        default_kwargs={"k": 1},
        reference="Thiede & Brahma, Analyzing Failures of CVAE, 2019",
        compatibility=_prediction_matches_ground_truth("predictions", "gt"),
    )
    reg.register(
        name="prediction_nll",
        fn=driving.prediction_nll,
        category="prediction",
        unit="nats",
        required_inputs=("predictions", "log_weights", "gt"),
        reference="Thiede & Brahma, NeurIPS Workshop 2019",
        compatibility=_prediction_matches_ground_truth("predictions", "gt"),
    )
    reg.register(
        name="displacement_at_k",
        fn=driving.displacement_at_k,
        category="prediction",
        unit="meters",
        required_inputs=("predictions", "gt"),
        default_kwargs={"k": 6},
        reference="Chang et al., Argoverse, CVPR 2019",
        compatibility=_prediction_matches_ground_truth("predictions", "gt"),
    )

    reg.register(
        name="acceleration",
        fn=comfort.acceleration,
        category="comfort",
        unit="m/s^2",
        required_inputs=("traj", "dt"),
        reference="Standard finite-difference kinematics",
        compatibility=_trajectory_input("traj"),
    )
    reg.register(
        name="jerk",
        fn=comfort.jerk,
        category="comfort",
        unit="m/s^3",
        required_inputs=("traj", "dt"),
        reference="Standard finite-difference kinematics",
        compatibility=_trajectory_input("traj"),
    )
    reg.register(
        name="jerk_cost",
        fn=comfort.jerk_cost,
        category="comfort",
        unit="m^2/s^6",
        required_inputs=("traj", "dt"),
        reference="RoboMetrics internal comfort cost",
        is_novel=True,
        compatibility=_trajectory_input("traj"),
    )
    reg.register(
        name="acceleration_magnitude",
        fn=comfort.acceleration_magnitude,
        category="comfort",
        unit="m/s^2",
        required_inputs=("traj", "dt"),
        reference="Standard finite-difference kinematics",
        compatibility=_trajectory_input("traj"),
    )
    reg.register(
        name="jerk_magnitude",
        fn=comfort.jerk_magnitude,
        category="comfort",
        unit="m/s^3",
        required_inputs=("traj", "dt"),
        reference="Standard finite-difference kinematics",
        compatibility=_trajectory_input("traj"),
    )
    reg.register(
        name="max_acceleration",
        fn=comfort.max_acceleration,
        category="comfort",
        unit="m/s^2",
        required_inputs=("traj", "dt"),
        reference="Standard peak acceleration evaluation",
        compatibility=_trajectory_input("traj"),
    )
    reg.register(
        name="mean_acceleration",
        fn=comfort.mean_acceleration,
        category="comfort",
        unit="m/s^2",
        required_inputs=("traj", "dt"),
        reference="Standard mean acceleration evaluation",
        compatibility=_trajectory_input("traj"),
    )
    reg.register(
        name="rms_acceleration",
        fn=comfort.rms_acceleration,
        category="comfort",
        unit="m/s^2",
        required_inputs=("traj", "dt"),
        reference="Standard RMS acceleration evaluation",
        compatibility=_trajectory_input("traj"),
    )
    reg.register(
        name="max_deceleration",
        fn=comfort.max_deceleration,
        category="comfort",
        unit="m/s^2",
        required_inputs=("traj", "dt"),
        reference="Standard peak deceleration evaluation",
        compatibility=_trajectory_input("traj"),
    )
    reg.register(
        name="smoothness_score",
        fn=comfort.smoothness_score,
        category="comfort",
        unit="score",
        required_inputs=("traj",),
        reference="RoboMetrics internal heuristic",
        is_novel=True,
        compatibility=_trajectory_input("traj"),
    )

    reg.register(
        name="collision_rate",
        fn=safety.collision_rate,
        category="safety",
        unit="ratio",
        required_inputs=("ego_traj", "actor_trajs", "ego_radius", "actor_radius"),
        reference="Standard disc-overlap collision metric for robotics simulation",
        compatibility=_trajectory_input("ego_traj"),
    )
    reg.register(
        name="collision_rate_obb",
        fn=safety.collision_rate_obb,
        category="safety",
        unit="ratio",
        required_inputs=(
            "ego_traj",
            "ego_dims",
            "ego_yaws",
            "actor_trajs",
            "actor_dims",
            "actor_yaws",
        ),
        reference="Gottschalk et al., OBBTree, SIGGRAPH 1996",
        compatibility=_trajectory_input("ego_traj"),
    )
    reg.register(
        name="time_to_collision",
        fn=safety.time_to_collision,
        category="safety",
        unit="seconds",
        required_inputs=("ego_state", "actor_state"),
        reference="Hayward, Time-to-collision, 1972",
        higher_is_better=True,
    )
    reg.register(
        name="min_distance_to_actors",
        fn=safety.min_distance_to_actors,
        category="safety",
        unit="meters",
        required_inputs=("ego_traj", "actor_trajs"),
        reference="Standard minimum Euclidean separation metric",
        higher_is_better=True,
        compatibility=_trajectory_input("ego_traj"),
    )
    reg.register(
        name="lane_departure_rate",
        fn=safety.lane_departure_rate,
        category="safety",
        unit="ratio",
        required_inputs=("ego_traj", "lane_boundary"),
        reference="Standard lane boundary containment metric",
        compatibility=_trajectory_input("ego_traj"),
    )
    reg.register(
        name="offroad_rate",
        fn=driving.offroad_rate,
        category="safety",
        unit="ratio",
        required_inputs=("ego_traj", "drivable_polygons"),
        reference="Caesar et al., nuScenes, CVPR 2020",
        higher_is_better=False,
        compatibility=_trajectory_input("ego_traj"),
    )
    reg.register(
        name="soft_ttc",
        fn=driving.soft_ttc,
        category="safety",
        unit="seconds",
        required_inputs=("ego_traj", "actor_trajs", "dt"),
        default_kwargs={"ego_radius": 0.0, "actor_radius": 0.0},
        reference="Weng et al., nuScenes-Forecast, ECCV 2022",
        higher_is_better=True,
        compatibility=_trajectory_input("ego_traj"),
    )

    reg.register(
        name="speed_profile",
        fn=physics.speed_profile,
        category="physics",
        unit="m/s",
        required_inputs=("traj", "dt"),
        reference="Standard finite-difference kinematics",
        compatibility=_trajectory_input("traj"),
    )
    reg.register(
        name="acceleration_limits_violated",
        fn=physics.acceleration_limits_violated,
        category="physics",
        unit="m/s^2",
        required_inputs=("traj", "dt", "max_accel"),
        reference="Standard acceleration constraint violation metric",
        compatibility=_trajectory_input("traj"),
    )
    reg.register(
        name="jerk_limits_violated",
        fn=physics.jerk_limits_violated,
        category="physics",
        unit="m/s^3",
        required_inputs=("traj", "dt", "max_jerk"),
        reference="Standard jerk constraint violation metric",
        compatibility=_trajectory_input("traj"),
    )
    reg.register(
        name="curvature_limits_violated",
        fn=physics.curvature_limits_violated,
        category="physics",
        unit="1/m",
        required_inputs=("traj", "max_curvature"),
        reference="Standard curvature constraint violation metric",
        compatibility=_trajectory_input("traj"),
    )
    reg.register(
        name="dynamic_feasibility_score",
        fn=physics.dynamic_feasibility_score,
        category="physics",
        unit="score",
        required_inputs=("traj", "dt"),
        default_kwargs={"constraints": {}},
        reference="RoboMetrics internal heuristic",
        is_novel=True,
        compatibility=_trajectory_input("traj"),
    )

    reg.register(
        name="compounding_error_index",
        fn=temporal.compounding_error_index,
        category="temporal",
        unit="ratio",
        required_inputs=("errors",),
        reference="RoboMetrics internal heuristic",
        is_novel=True,
    )
    reg.register(
        name="workspace_coverage",
        fn=coverage.workspace_coverage,
        category="coverage",
        unit="cells",
        required_inputs=("points",),
        default_kwargs={"cell_size": 1.0},
        reference="Standard grid-cell workspace coverage metric",
        higher_is_better=True,
    )
    reg.register(
        name="calibration_error",
        fn=calibration.calibration_error,
        category="calibration",
        unit="ratio",
        required_inputs=("confidences", "outcomes"),
        default_kwargs={"n_bins": 10},
        reference="Naeini et al., ECE, AAAI 2015",
    )
    reg.register(
        name="trajectory_diversity",
        fn=diversity.trajectory_diversity,
        category="diversity",
        unit="meters",
        required_inputs=("predictions",),
        reference="Standard pairwise trajectory diversity metric",
    )

    reg.register(
        name="task_success_rate",
        fn=task.task_success_rate,
        category="task",
        unit="ratio",
        required_inputs=("outcomes",),
        reference="Standard binary task evaluation used in robotics benchmarks",
    )
    reg.register(
        name="goal_reaching_accuracy",
        fn=task.goal_reaching_accuracy,
        category="task",
        unit="ratio",
        required_inputs=("positions", "goals", "tolerance"),
        default_kwargs={"tolerance": 1.0},
        reference=(
            "Anderson et al., Habitat: A Platform for Embodied AI Research, ICCV 2019"
        ),
    )
    reg.register(
        name="grasp_success_rate",
        fn=manipulation.grasp_success_rate,
        category="task",
        unit="ratio",
        required_inputs=("attempts", "successes"),
        reference="Mahler et al., Dex-Net 2.0, RSS 2017",
    )
    reg.register(
        name="contact_richness",
        fn=manipulation.contact_richness,
        category="task",
        unit="score",
        required_inputs=("contact_forces",),
        default_kwargs={"threshold": 0.1},
        reference="Handa et al., DexPilot, ICRA 2020",
        higher_is_better=True,
    )
    reg.register(
        name="force_limit_compliance",
        fn=manipulation.force_limit_compliance,
        category="task",
        unit="score",
        required_inputs=("forces", "max_force"),
        reference="ISO/TS 15066 collaborative robot safety",
        higher_is_better=True,
    )
    reg.register(
        name="joint_limit_violation_rate",
        fn=manipulation.joint_limit_violation_rate,
        category="task",
        unit="ratio",
        required_inputs=("joint_angles", "lower_limits", "upper_limits"),
        reference="Siciliano et al., Robotics, Springer 2009",
        higher_is_better=False,
    )
    reg.register(
        name="end_effector_tracking_error",
        fn=manipulation.end_effector_tracking_error,
        category="task",
        unit="meters",
        required_inputs=("ee_traj", "target_traj"),
        compatibility=_same_shape("ee_traj", "target_traj"),
        reference="Siciliano et al., Robotics, Springer 2009",
    )

    return reg


def load_pack(
    module_name: str,
    registry: Optional[MetricRegistry] = None,
) -> list[MetricDefinition]:
    """Import a metric pack module and register its metrics.

    The module must expose a module-level list named `METRIC_PACK` where
    each element is a dict compatible with MetricRegistry.register_many().

    Example pack module:

        METRIC_PACK = [
            {
                "name": "my_custom_metric",
                "fn": my_custom_metric_fn,
                "category": "custom",
                "unit": "score",
                "reference": "My Paper, 2024",
            }
        ]

    Raises ImportError if the module cannot be imported.
    Raises ValueError if the module does not expose METRIC_PACK.
    Pass ``registry=custom_registry`` to load the pack into an isolated
    ``MetricRegistry`` used by a custom ``Evaluator``. By default, packs are
    registered in the global registry.

    Returns the list of registered MetricDefinition objects.
    """
    module = importlib.import_module(module_name)
    if not hasattr(module, "METRIC_PACK"):
        raise ValueError(f"{module_name} does not expose METRIC_PACK")
    pack = module.METRIC_PACK
    if not isinstance(pack, list):
        raise ValueError(f"{module_name}.METRIC_PACK must be a list")
    target_registry = globals()["registry"] if registry is None else registry
    return target_registry.register_many(pack)


_LOWER_IS_BETTER_RATE_NAMES = {
    "collision_rate",
    "lane_departure_rate",
    "miss_rate",
    "near_miss_rate",
    "offroad_rate",
    "physics_violation_rate",
    "joint_limit_violation_rate",
}


def _default_higher_is_better(name: str) -> bool:
    if name in _LOWER_IS_BETTER_RATE_NAMES:
        return False
    suffixes = (
        "_score",
        "_rate",
        "_accuracy",
        "_diversity",
        "_smoothness",
        "success_rate",
        "feasibility",
    )
    return any(name.endswith(suffix) or name == suffix for suffix in suffixes)


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _first_doc_line(fn: MetricFn) -> str:
    doc = getattr(fn, "__doc__", None)
    if not isinstance(doc, str) or not doc:
        return ""
    return doc.strip().splitlines()[0]


def _shape(value: Any) -> Optional[tuple[int, ...]]:
    try:
        return tuple(np.asarray(value).shape)
    except (TypeError, ValueError):
        return None


def _is_trajectory_shape(shape: Optional[tuple[int, ...]]) -> bool:
    return shape is not None and len(shape) == 2 and shape[0] > 0 and shape[1] in (2, 3)


def _is_prediction_shape(shape: Optional[tuple[int, ...]]) -> bool:
    return (
        shape is not None
        and len(shape) == 3
        and shape[0] > 0
        and shape[1] > 0
        and shape[2] in (2, 3)
    )


def _trajectory_input(key: str) -> CompatibilityFn:
    def compatible(inputs: Mapping[str, Any]) -> bool:
        return _is_trajectory_shape(_shape(inputs[key]))

    return compatible


def _same_shape(left: str, right: str) -> CompatibilityFn:
    def compatible(inputs: Mapping[str, Any]) -> bool:
        left_shape = _shape(inputs[left])
        right_shape = _shape(inputs[right])
        return _is_trajectory_shape(left_shape) and left_shape == right_shape

    return compatible


def _same_dimensionality(left: str, right: str) -> CompatibilityFn:
    def compatible(inputs: Mapping[str, Any]) -> bool:
        left_shape = _shape(inputs[left])
        right_shape = _shape(inputs[right])
        return (
            _is_trajectory_shape(left_shape)
            and _is_trajectory_shape(right_shape)
            and left_shape is not None
            and right_shape is not None
            and left_shape[1] == right_shape[1]
        )

    return compatible


def _prediction_matches_ground_truth(predictions_key: str, gt_key: str) -> CompatibilityFn:
    def compatible(inputs: Mapping[str, Any]) -> bool:
        prediction_shape = _shape(inputs[predictions_key])
        gt_shape = _shape(inputs[gt_key])
        return (
            _is_prediction_shape(prediction_shape)
            and _is_trajectory_shape(gt_shape)
            and prediction_shape is not None
            and gt_shape is not None
            and prediction_shape[1:] == gt_shape
        )

    return compatible


registry = create_default_registry()

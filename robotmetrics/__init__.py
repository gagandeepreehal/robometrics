"""RobotMetrics: lightweight metrics for Physical AI systems."""

from robotmetrics.comfort import (
    acceleration,
    jerk,
    jerk_cost,
    max_acceleration,
    max_deceleration,
    smoothness_score,
)
from robotmetrics.evaluator import EvaluationInputError, Evaluator
from robotmetrics.physics import (
    acceleration_limits_violated,
    curvature_limits_violated,
    dynamic_feasibility_score,
    jerk_limits_violated,
    speed_profile,
)
from robotmetrics.prediction import min_ade, min_fde, miss_rate, topk_trajectory_error
from robotmetrics.registry import MetricDefinition, MetricRegistry, UnknownMetricError, registry
from robotmetrics.results import EvaluationResult, MetricResult
from robotmetrics.safety import (
    collision_rate,
    lane_departure_rate,
    min_distance_to_actors,
    time_to_collision,
)
from robotmetrics.schemas import AgentState, Trajectory
from robotmetrics.trajectory import (
    average_displacement_error,
    curvature,
    final_displacement_error,
    hausdorff_distance,
    lateral_error,
    longitudinal_error,
    path_length,
)

__all__ = [
    "AgentState",
    "EvaluationInputError",
    "EvaluationResult",
    "Evaluator",
    "MetricDefinition",
    "MetricRegistry",
    "MetricResult",
    "Trajectory",
    "UnknownMetricError",
    "acceleration",
    "acceleration_limits_violated",
    "average_displacement_error",
    "collision_rate",
    "curvature",
    "curvature_limits_violated",
    "dynamic_feasibility_score",
    "final_displacement_error",
    "hausdorff_distance",
    "jerk",
    "jerk_cost",
    "jerk_limits_violated",
    "lane_departure_rate",
    "lateral_error",
    "longitudinal_error",
    "max_acceleration",
    "max_deceleration",
    "min_ade",
    "min_distance_to_actors",
    "min_fde",
    "miss_rate",
    "path_length",
    "registry",
    "smoothness_score",
    "speed_profile",
    "time_to_collision",
    "topk_trajectory_error",
]

__version__ = "0.1.0"

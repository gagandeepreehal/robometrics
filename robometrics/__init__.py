"""RoboMetrics: lightweight robotics metrics for Python."""

from robometrics._version import __version__
from robometrics.comfort import (
    acceleration,
    acceleration_magnitude,
    jerk,
    jerk_cost,
    jerk_magnitude,
    max_acceleration,
    max_deceleration,
    mean_acceleration,
    rms_acceleration,
    smoothness_score,
)
from robometrics.evaluator import EvaluationInputError, Evaluator
from robometrics.io import TrajectoryIOError
from robometrics.physics import (
    acceleration_limits_violated,
    curvature_limits_violated,
    dynamic_feasibility_score,
    jerk_limits_violated,
    speed_profile,
)
from robometrics.prediction import min_ade, min_fde, miss_rate, topk_trajectory_error
from robometrics.registry import MetricDefinition, MetricRegistry, UnknownMetricError, registry
from robometrics.results import ComparisonResult, EvaluationResult, MetricComparison, MetricResult
from robometrics.safety import (
    collision_rate,
    lane_departure_rate,
    min_distance_to_actors,
    time_to_collision,
)
from robometrics.schemas import AgentState, Trajectory
from robometrics.task import goal_reaching_accuracy, task_success_rate
from robometrics.trajectory import (
    average_displacement_error,
    curvature,
    curvature_profile,
    final_displacement_error,
    hausdorff_distance,
    lateral_error,
    longitudinal_error,
    mean_curvature,
    path_length,
)

ade = average_displacement_error
fde = final_displacement_error

__all__ = [
    "AgentState",
    "ComparisonResult",
    "EvaluationInputError",
    "EvaluationResult",
    "Evaluator",
    "MetricDefinition",
    "MetricComparison",
    "MetricRegistry",
    "MetricResult",
    "Trajectory",
    "TrajectoryIOError",
    "UnknownMetricError",
    "__version__",
    "acceleration",
    "acceleration_magnitude",
    "acceleration_limits_violated",
    "ade",
    "average_displacement_error",
    "collision_rate",
    "curvature",
    "curvature_profile",
    "curvature_limits_violated",
    "dynamic_feasibility_score",
    "fde",
    "final_displacement_error",
    "goal_reaching_accuracy",
    "hausdorff_distance",
    "jerk",
    "jerk_cost",
    "jerk_magnitude",
    "jerk_limits_violated",
    "lane_departure_rate",
    "lateral_error",
    "longitudinal_error",
    "max_acceleration",
    "max_deceleration",
    "mean_acceleration",
    "mean_curvature",
    "min_ade",
    "min_distance_to_actors",
    "min_fde",
    "miss_rate",
    "path_length",
    "registry",
    "rms_acceleration",
    "smoothness_score",
    "speed_profile",
    "task_success_rate",
    "time_to_collision",
    "topk_trajectory_error",
]

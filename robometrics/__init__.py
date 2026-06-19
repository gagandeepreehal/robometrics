"""RoboMetrics: lightweight robotics metrics for Python."""

from robometrics._version import __version__
from robometrics.calibration import calibration_error
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
from robometrics.coverage import coverage_score
from robometrics.diversity import behavioral_diversity
from robometrics.evaluator import EvaluationInputError, Evaluator
from robometrics.io import TrajectoryIOError
from robometrics.physics import (
    acceleration_limits_violated,
    curvature_limits_violated,
    dynamic_feasibility,
    dynamic_feasibility_score,
    jerk_limits_violated,
    kinematic_feasibility,
    physics_violation_rate,
    speed_profile,
)
from robometrics.prediction import min_ade, min_fde, miss_rate, topk_trajectory_error
from robometrics.registry import MetricDefinition, MetricRegistry, UnknownMetricError, registry
from robometrics.results import EvaluationResult, MetricResult
from robometrics.safety import (
    collision_rate,
    failure_severity,
    intervention_free_time,
    lane_departure_rate,
    min_distance_to_actors,
    near_miss_rate,
    recovery_success_rate,
    time_to_collision,
)
from robometrics.schemas import AgentState, Trajectory
from robometrics.temporal import (
    action_jerk,
    compounding_error_index,
    control_smoothness,
    long_horizon_drift,
    temporal_drift,
)
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
    "EvaluationInputError",
    "EvaluationResult",
    "Evaluator",
    "MetricDefinition",
    "MetricRegistry",
    "MetricResult",
    "Trajectory",
    "TrajectoryIOError",
    "UnknownMetricError",
    "__version__",
    "acceleration",
    "acceleration_magnitude",
    "acceleration_limits_violated",
    "action_jerk",
    "ade",
    "average_displacement_error",
    "behavioral_diversity",
    "calibration_error",
    "collision_rate",
    "compounding_error_index",
    "control_smoothness",
    "coverage_score",
    "curvature",
    "curvature_profile",
    "curvature_limits_violated",
    "dynamic_feasibility",
    "dynamic_feasibility_score",
    "failure_severity",
    "fde",
    "final_displacement_error",
    "hausdorff_distance",
    "intervention_free_time",
    "jerk",
    "jerk_cost",
    "jerk_magnitude",
    "jerk_limits_violated",
    "kinematic_feasibility",
    "lane_departure_rate",
    "lateral_error",
    "long_horizon_drift",
    "longitudinal_error",
    "max_acceleration",
    "max_deceleration",
    "mean_acceleration",
    "mean_curvature",
    "min_ade",
    "min_distance_to_actors",
    "min_fde",
    "miss_rate",
    "near_miss_rate",
    "path_length",
    "physics_violation_rate",
    "recovery_success_rate",
    "registry",
    "rms_acceleration",
    "smoothness_score",
    "speed_profile",
    "temporal_drift",
    "time_to_collision",
    "topk_trajectory_error",
]

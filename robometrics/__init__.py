"""RoboMetrics: lightweight robotics metrics for Python."""

from robometrics._version import __version__
from robometrics.accumulator import MetricAccumulator
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
from robometrics.coverage import workspace_coverage
from robometrics.diversity import trajectory_diversity
from robometrics.driving import (
    displacement_at_k,
    offroad_rate,
    prediction_nll,
    soft_ttc,
)
from robometrics.evaluator import EvaluationInputError, Evaluator
from robometrics.history import CheckpointEntry, EvaluationHistory
from robometrics.io import TrajectoryIOError
from robometrics.manipulation import (
    contact_richness,
    end_effector_tracking_error,
    force_limit_compliance,
    grasp_success_rate,
    joint_limit_violation_rate,
)
from robometrics.physics import (
    acceleration_limits_violated,
    curvature_limits_violated,
    dynamic_feasibility_score,
    jerk_limits_violated,
    speed_profile,
)
from robometrics.prediction import min_ade, min_fde, miss_rate, topk_trajectory_error
from robometrics.registry import (
    MetricDefinition,
    MetricRegistry,
    UnknownMetricError,
    load_pack,
    registry,
)
from robometrics.results import ComparisonResult, EvaluationResult, MetricComparison, MetricResult
from robometrics.safety import (
    collision_rate,
    collision_rate_obb,
    lane_departure_rate,
    min_distance_to_actors,
    time_to_collision,
)
from robometrics.schemas import AgentState, Trajectory
from robometrics.task import goal_reaching_accuracy, task_success_rate
from robometrics.temporal import compounding_error_index
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
    "CheckpointEntry",
    "ComparisonResult",
    "EvaluationInputError",
    "EvaluationHistory",
    "EvaluationResult",
    "Evaluator",
    "MetricDefinition",
    "MetricComparison",
    "MetricAccumulator",
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
    "calibration_error",
    "collision_rate",
    "collision_rate_obb",
    "compounding_error_index",
    "contact_richness",
    "curvature",
    "curvature_profile",
    "curvature_limits_violated",
    "displacement_at_k",
    "dynamic_feasibility_score",
    "end_effector_tracking_error",
    "fde",
    "final_displacement_error",
    "force_limit_compliance",
    "goal_reaching_accuracy",
    "grasp_success_rate",
    "hausdorff_distance",
    "jerk",
    "jerk_cost",
    "jerk_magnitude",
    "jerk_limits_violated",
    "joint_limit_violation_rate",
    "lane_departure_rate",
    "lateral_error",
    "load_pack",
    "longitudinal_error",
    "max_acceleration",
    "max_deceleration",
    "mean_acceleration",
    "mean_curvature",
    "min_ade",
    "min_distance_to_actors",
    "min_fde",
    "miss_rate",
    "offroad_rate",
    "path_length",
    "prediction_nll",
    "registry",
    "rms_acceleration",
    "smoothness_score",
    "soft_ttc",
    "speed_profile",
    "task_success_rate",
    "time_to_collision",
    "topk_trajectory_error",
    "trajectory_diversity",
    "workspace_coverage",
]

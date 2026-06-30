# API

This page lists every public symbol exported by `robometrics.__all__`.

| Symbol | Module | Description |
| --- | --- | --- |
| `AgentState` | `robometrics.schemas` | Planar constant-velocity agent state. |
| `BenchmarkProfile` | `robometrics.benchmarks` | Declarative benchmark profile metadata. |
| `CheckpointEntry` | `robometrics.history` | One recorded evaluation checkpoint. |
| `ComparisonResult` | `robometrics.results` | Metric-by-metric comparison between two evaluation results. |
| `DatasetValidationResult` | `robometrics.validation` | Validation report for one CSV or JSON trajectory-style dataset. |
| `EvaluationInputError` | `robometrics.evaluator` | Raised when an evaluation request cannot be constructed. |
| `EvaluationHistory` | `robometrics.history` | Track EvaluationResult objects across training checkpoints. |
| `EvaluationResult` | `robometrics.results` | Collection of metric results from one local evaluation run. |
| `EVALUATION_RESULT_SCHEMA_VERSION` | `robometrics.results` | Stable `EvaluationResult` JSON schema version string. |
| `Evaluator` | `robometrics.evaluator` | Evaluate registered RoboMetrics metrics against local Python inputs, including `Trajectory` schema objects. |
| `MetricDefinition` | `robometrics.registry` | Registered metric metadata and execution contract. |
| `MetricComparison` | `robometrics.results` | Comparison of one metric across two evaluation results. |
| `MetricAccumulator` | `robometrics.accumulator` | Stateful per-metric accumulator for streaming evaluation. |
| `MetricRegistry` | `robometrics.registry` | Registry for named metric callables. |
| `MetricResult` | `robometrics.results` | Result for one metric execution. |
| `Trajectory` | `robometrics.schemas` | Serializable trajectory with optional timestamps. |
| `TrajectoryIOError` | `robometrics.io` | Raised when a trajectory file cannot be read or parsed. |
| `UnknownMetricError` | `robometrics.registry` | Raised when a metric name is not registered. |
| `ValidationIssue` | `robometrics.validation` | One dataset validation issue. |
| `__version__` | `robometrics._version` | Package version string. |
| `acceleration` | `robometrics.comfort` | Return approximate acceleration vectors from a position trajectory. |
| `acceleration_magnitude` | `robometrics.comfort` | Return per-step acceleration magnitudes. |
| `acceleration_limits_violated` | `robometrics.physics` | Return a MetricResult describing whether acceleration exceeds max_accel. |
| `action_jerk` | `robometrics.temporal` | Return mean squared second finite difference of an action sequence. |
| `ade` | `robometrics.trajectory` | Return mean pointwise Euclidean distance between predicted and ground-truth paths. |
| `average_displacement_error` | `robometrics.trajectory` | Return mean pointwise Euclidean distance between predicted and ground-truth paths. |
| `behavioral_diversity` | `robometrics.diversity` | Return mean pairwise distance between unique behavior samples. |
| `calibration_error` | `robometrics.calibration` | Return expected calibration error for binary outcomes and confidences. |
| `collision_rate` | `robometrics.safety` | Return fraction of actor-covered ego timesteps that collide with at least one actor. |
| `collision_rate_obb` | `robometrics.safety` | Return fraction of actor-covered ego timesteps with OBB collision. |
| `compounding_error_index` | `robometrics.temporal` | Return normalized non-negative temporal error growth. |
| `contact_richness` | `robometrics.manipulation` | Return fraction of timesteps with meaningful contact force magnitude. |
| `control_smoothness` | `robometrics.temporal` | Return a bounded control smoothness score. |
| `coverage_score` | `robometrics.coverage` | Return occupied grid-bin coverage as a score in [0, 1]. |
| `curvature` | `robometrics.trajectory` | Return approximate planar XY curvature assuming uniformly spaced samples. |
| `curvature_profile` | `robometrics.trajectory` | Return approximate planar XY curvature assuming uniformly spaced samples. |
| `curvature_limits_violated` | `robometrics.physics` | Return a MetricResult describing whether curvature exceeds max_curvature. |
| `displacement_at_k` | `robometrics.driving` | Return mean displacement error using only the top-K predicted modes. |
| `dynamic_feasibility` | `robometrics.physics` | Return a Newtonian dynamic feasibility score. |
| `dynamic_feasibility_score` | `robometrics.physics` | Return a conservative 0..1 feasibility score against dynamic limits. |
| `end_effector_tracking_error` | `robometrics.manipulation` | Return mean Euclidean distance between end-effector and target path. |
| `failure_severity` | `robometrics.safety` | Return aggregated numeric severity for failure events. |
| `fde` | `robometrics.trajectory` | Return final-step Euclidean distance between predicted and ground-truth paths. |
| `final_displacement_error` | `robometrics.trajectory` | Return final-step Euclidean distance between predicted and ground-truth paths. |
| `force_limit_compliance` | `robometrics.manipulation` | Return fraction of timesteps where force magnitude is within limit. |
| `get_profile` | `robometrics.benchmarks` | Return one benchmark profile by name. |
| `goal_reaching_accuracy` | `robometrics.task` | Return fraction of goals reached within Euclidean tolerance. |
| `grasp_success_rate` | `robometrics.manipulation` | Return fraction of grasp attempts that succeeded. |
| `hausdorff_distance` | `robometrics.trajectory` | Return symmetric Hausdorff distance using all coordinate dimensions. |
| `intervention_free_time` | `robometrics.safety` | Return duration of intervention-free segments. |
| `jerk` | `robometrics.comfort` | Return approximate jerk vectors from a position trajectory. |
| `jerk_cost` | `robometrics.comfort` | Return mean squared jerk magnitude. |
| `jerk_magnitude` | `robometrics.comfort` | Return per-step jerk magnitudes. |
| `jerk_limits_violated` | `robometrics.physics` | Return a MetricResult describing whether jerk exceeds max_jerk. |
| `joint_limit_violation_rate` | `robometrics.manipulation` | Return fraction of configs where any joint exceeds its limits. |
| `kinematic_feasibility` | `robometrics.physics` | Return a trajectory kinematic feasibility score. |
| `lane_departure_rate` | `robometrics.safety` | Return fraction of ego points outside a polygonal lane boundary. |
| `lateral_error` | `robometrics.trajectory` | Return mean absolute planar XY lateral deviation from a reference trajectory. |
| `list_profiles` | `robometrics.benchmarks` | Return all built-in benchmark profiles. |
| `load_pack` | `robometrics.registry` | Import a metric pack module and register its metrics. |
| `load_trajectory` | `robometrics.io` | Load a trajectory from .npy, .npz, .csv, or .json. |
| `long_horizon_drift` | `robometrics.temporal` | Return later-weighted mean rollout error. |
| `longitudinal_error` | `robometrics.trajectory` | Return mean absolute planar XY longitudinal deviation along a reference trajectory. |
| `max_acceleration` | `robometrics.comfort` | Return maximum acceleration magnitude. |
| `max_deceleration` | `robometrics.comfort` | Return maximum longitudinal deceleration magnitude. |
| `mean_acceleration` | `robometrics.comfort` | Return mean acceleration magnitude. |
| `mean_curvature` | `robometrics.trajectory` | Return mean planar XY curvature. |
| `min_ade` | `robometrics.prediction` | Return the minimum average displacement error over predicted modes. |
| `min_distance_to_actors` | `robometrics.safety` | Return minimum time-aligned Euclidean distance from ego to any actor. |
| `min_fde` | `robometrics.prediction` | Return the minimum final displacement error over predicted modes. |
| `miss_rate` | `robometrics.prediction` | Return a per-sample miss indicator as 0.0 or 1.0. |
| `near_miss_rate` | `robometrics.safety` | Return fraction of events that are near misses without collision. |
| `offroad_rate` | `robometrics.driving` | Return fraction of ego positions outside all drivable-area polygons. |
| `path_length` | `robometrics.trajectory` | Return total Euclidean path length. |
| `physics_violation_rate` | `robometrics.physics` | Return fraction of timesteps with any physics violation. |
| `prediction_nll` | `robometrics.driving` | Return mean negative log-likelihood for a Gaussian mixture prediction. |
| `recovery_success_rate` | `robometrics.safety` | Return successful recoveries divided by recovery opportunities. |
| `registry` | `robometrics.registry` | Registry for named metric callables. |
| `rms_acceleration` | `robometrics.comfort` | Return root-mean-square acceleration magnitude. |
| `run_profile` | `robometrics.benchmarks` | Run a benchmark profile against prediction and ground-truth arrays. |
| `smoothness_score` | `robometrics.comfort` | Return a scale-normalized third-difference smoothness score. |
| `soft_ttc` | `robometrics.driving` | Return minimum constant-velocity TTC computed at each rollout timestep. |
| `speed_profile` | `robometrics.physics` | Return speed magnitude at each trajectory point. |
| `task_success_rate` | `robometrics.task` | Return fraction of task attempts that succeeded. |
| `temporal_drift` | `robometrics.temporal` | Return mean slope of L2 error over time. |
| `time_to_collision` | `robometrics.safety` | Return constant-velocity time to collision for two disc agents. |
| `topk_trajectory_error` | `robometrics.prediction` | Return best ADE among the first k predictions, assumed confidence-ranked. |
| `trajectory_diversity` | `robometrics.diversity` | Return mean pairwise ADE between predicted trajectory modes. |
| `validate_dataset` | `robometrics.validation` | Validate a trajectory-style CSV or JSON file. |
| `workspace_coverage` | `robometrics.coverage` | Return the number of occupied grid cells visited by sampled positions. |

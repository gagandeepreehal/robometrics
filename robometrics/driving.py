"""Autonomous driving metric pack."""

from __future__ import annotations

from math import inf

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp

from robometrics.geometry import (
    FloatArray,
    as_actor_trajectories,
    as_prediction_set,
    as_trajectory,
    points_in_polygon,
    require_same_time_and_dim,
    validate_nonnegative,
    validate_positive,
    xy,
)
from robometrics.prediction import min_ade
from robometrics.safety import _solve_ttc_quadratic


def prediction_nll(
    predictions: ArrayLike,
    log_weights: ArrayLike,
    ground_truth: ArrayLike,
) -> float:
    """Return mean negative log-likelihood for a Gaussian mixture prediction.

    Formula:
        For K modes with log-weights log_w[k] and Gaussian mean predictions[k],
        compute the log-sum-exp of (log_w[k] - 0.5 * ||pred[k] - gt||^2 / sigma^2)
        assuming unit variance (sigma=1). Return the negative mean over timesteps.
        This is a simplified NLL; for full covariance NLL supply sigma explicitly.

    Reference: Thiede & Brahma, Analyzing Failures of Conditional Variational
               Autoencoders, NeurIPS Workshop 2019.

    Inputs:
        predictions: KxTx2 or KxTx3 array of K predicted trajectories.
        log_weights: K-length array of log mixture weights (need not be normalized).
        ground_truth: Tx2 or Tx3 ground-truth trajectory.

    Output:
        A scalar NLL where lower is better. Returns inf if predictions is empty.
    """
    pred_arr = as_prediction_set(predictions)
    gt_arr = as_trajectory(ground_truth, name="ground_truth")
    require_same_time_and_dim(pred_arr, gt_arr, ground_truth_name="ground_truth")
    weights = _as_log_weights(log_weights, expected_modes=pred_arr.shape[0])

    squared_distances = np.sum((pred_arr - gt_arr[None, :, :]) ** 2, axis=2)
    timestep_log_likelihoods = logsumexp(weights[:, None] - 0.5 * squared_distances, axis=0)
    return float(-np.mean(timestep_log_likelihoods))


def offroad_rate(
    ego_traj: ArrayLike,
    drivable_polygons: list[ArrayLike],
) -> float:
    """Return fraction of ego positions outside all drivable-area polygons.

    Formula:
        For each ego position, check if it falls inside any polygon using the
        existing `points_in_polygon` helper from `robometrics.geometry`.
        A position is on-road if it is inside at least one polygon.
        The metric is mean(~on_road_mask).

    Reference: Caesar et al., nuScenes: A Multimodal Dataset for Autonomous
               Driving, CVPR 2020.

    Inputs:
        ego_traj: Nx2 or Nx3 ego trajectory.
        drivable_polygons: list of Mx2 polygon vertex arrays defining
                           drivable areas. Empty list -> all positions off-road.

    Output:
        A rate in [0, 1] where 0.0 means always on-road.
    """
    ego = xy(as_trajectory(ego_traj, name="ego_traj"))
    if not drivable_polygons:
        return 1.0

    on_road = np.zeros(ego.shape[0], dtype=np.bool_)
    for index, polygon in enumerate(drivable_polygons):
        polygon_arr = xy(as_trajectory(polygon, name=f"drivable_polygons[{index}]"))
        if polygon_arr.shape[0] < 3:
            raise ValueError("drivable_polygons must contain polygons with at least 3 vertices")
        on_road |= points_in_polygon(ego, polygon_arr)
    return float(np.mean(~on_road))


def soft_ttc(
    ego_traj: ArrayLike,
    actor_trajs: object,
    dt: float,
    *,
    ego_radius: float = 0.0,
    actor_radius: float = 0.0,
) -> float:
    """Return minimum constant-velocity TTC computed at each rollout timestep.

    Formula:
        At each timestep t, compute instantaneous velocity as the finite
        difference of position. Then run the constant-velocity TTC formula
        (quadratic solve as in `time_to_collision`) between ego at time t and
        each actor at time t. Return the minimum TTC across all timesteps and
        actors. math.inf means no collision predicted in the rollout.

    Reference: Weng et al., nuScenes-Forecast: Trajectory Forecasting in the
               Wild, ECCV 2022.

    Inputs:
        ego_traj: Nx2 ego trajectory (XY only).
        actor_trajs: list of Mx2 actor trajectories.
        dt: seconds per timestep (positive).
        ego_radius: collision disc radius for ego.
        actor_radius: collision disc radius for actors.

    Output:
        Minimum predicted TTC in seconds. math.inf if no collision predicted.
    """
    ego = _as_xy_rollout(ego_traj, name="ego_traj")
    actors = [
        _as_xy_rollout(actor, name=f"actor_trajs[{index}]")
        for index, actor in enumerate(as_actor_trajectories(actor_trajs))
    ]
    dt_value = validate_positive(float(dt), name="dt")
    ego_r = validate_nonnegative(float(ego_radius), name="ego_radius")
    actor_r = validate_nonnegative(float(actor_radius), name="actor_radius")
    if not actors:
        return inf

    ego_velocity = np.gradient(ego, dt_value, axis=0, edge_order=2)
    minimum = inf
    for actor in actors:
        actor_velocity = np.gradient(actor, dt_value, axis=0, edge_order=2)
        overlap = min(ego.shape[0], actor.shape[0])
        for step in range(overlap):
            ttc = _solve_ttc_quadratic(
                actor[step] - ego[step],
                actor_velocity[step] - ego_velocity[step],
                ego_r + actor_r,
            )
            minimum = min(minimum, ttc)
    return float(minimum)


def displacement_at_k(
    predictions: ArrayLike,
    ground_truth: ArrayLike,
    k: int,
) -> float:
    """Return mean displacement error using only the top-K predicted modes.

    Formula:
        Take the first K modes from predictions (assumed ranked by confidence,
        highest first). For each of the K modes compute ADE against ground_truth.
        Return the minimum ADE among those K modes (best-of-K).

    Reference: Chang et al., Argoverse: 3D Tracking and Forecasting with
               Rich Maps, CVPR 2019.

    Inputs:
        predictions: KxTxD array, modes ranked by confidence descending.
        ground_truth: TxD ground-truth trajectory.
        k: positive integer, number of top modes to consider. Clamped to K.

    Output:
        Minimum ADE over top-K modes in position units.
    """
    pred_arr = as_prediction_set(predictions)
    gt_arr = as_trajectory(ground_truth, name="ground_truth")
    require_same_time_and_dim(pred_arr, gt_arr, ground_truth_name="ground_truth")
    if k <= 0:
        raise ValueError("k must be positive")
    top_k = min(int(k), pred_arr.shape[0])
    return float(min_ade(pred_arr[:top_k], gt_arr))


def _as_log_weights(log_weights: ArrayLike, *, expected_modes: int) -> FloatArray:
    weights = np.asarray(log_weights, dtype=np.float64)
    if weights.ndim != 1:
        raise ValueError("log_weights must be a K-length array")
    if weights.shape[0] != expected_modes:
        raise ValueError("log_weights length must match predictions modes")
    if not np.all(np.isfinite(weights)):
        raise ValueError("log_weights must contain only finite values")
    return weights


def _as_xy_rollout(data: ArrayLike, *, name: str) -> FloatArray:
    arr = as_trajectory(data, name=name)
    if arr.shape[1] != 2:
        raise ValueError(f"{name} must be an Nx2 array")
    if arr.shape[0] < 3:
        raise ValueError(f"{name} must contain at least three points")
    return arr

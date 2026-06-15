"""Trajectory evaluation metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import directed_hausdorff

from robometrics.geometry import (
    FloatArray,
    as_trajectory,
    pointwise_distances,
    require_same_shape,
    vector_norms,
    xy,
)


def average_displacement_error(pred: ArrayLike, gt: ArrayLike) -> float:
    """Return mean pointwise Euclidean distance between predicted and ground-truth paths."""
    pred_arr = as_trajectory(pred, name="pred")
    gt_arr = as_trajectory(gt, name="gt")
    require_same_shape(pred_arr, gt_arr, "pred", "gt")
    return float(np.mean(pointwise_distances(pred_arr, gt_arr)))


def final_displacement_error(pred: ArrayLike, gt: ArrayLike) -> float:
    """Return final-step Euclidean distance between predicted and ground-truth paths."""
    pred_arr = as_trajectory(pred, name="pred")
    gt_arr = as_trajectory(gt, name="gt")
    require_same_shape(pred_arr, gt_arr, "pred", "gt")
    return float(np.linalg.norm(pred_arr[-1] - gt_arr[-1]))


def hausdorff_distance(pred: ArrayLike, gt: ArrayLike) -> float:
    """Return the symmetric Hausdorff distance between two trajectories."""
    pred_arr = as_trajectory(pred, name="pred")
    gt_arr = as_trajectory(gt, name="gt")
    if pred_arr.shape[1] != gt_arr.shape[1]:
        raise ValueError("pred and gt must have the same dimensionality")
    forward = float(directed_hausdorff(pred_arr, gt_arr)[0])
    backward = float(directed_hausdorff(gt_arr, pred_arr)[0])
    return max(forward, backward)


def path_length(traj: ArrayLike) -> float:
    """Return total Euclidean path length."""
    traj_arr = as_trajectory(traj, name="traj")
    if traj_arr.shape[0] == 1:
        return 0.0
    return float(np.sum(vector_norms(np.diff(traj_arr, axis=0))))


def curvature(traj: ArrayLike) -> FloatArray:
    """Return approximate planar curvature at each trajectory point."""
    traj_arr = as_trajectory(traj, name="traj")
    points = xy(traj_arr)
    if points.shape[0] < 3:
        return np.zeros(points.shape[0], dtype=np.float64)

    dx = np.gradient(points[:, 0], edge_order=2)
    dy = np.gradient(points[:, 1], edge_order=2)
    ddx = np.gradient(dx, edge_order=2)
    ddy = np.gradient(dy, edge_order=2)
    denominator = np.power(dx * dx + dy * dy, 1.5)
    numerator = np.abs(dx * ddy - dy * ddx)
    values = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 1e-12,
    )
    return np.asarray(values, dtype=np.float64)


def lateral_error(pred: ArrayLike, ref: ArrayLike) -> float:
    """Return mean absolute lateral deviation from a reference trajectory."""
    pred_arr = as_trajectory(pred, name="pred")
    ref_arr = as_trajectory(ref, name="ref")
    require_same_shape(pred_arr, ref_arr, "pred", "ref")
    tangents = _reference_tangents(ref_arr)
    deltas = xy(pred_arr) - xy(ref_arr)
    lateral = tangents[:, 0] * deltas[:, 1] - tangents[:, 1] * deltas[:, 0]
    zero_tangent = vector_norms(tangents) <= 1e-12
    lateral[zero_tangent] = vector_norms(deltas[zero_tangent])
    return float(np.mean(np.abs(lateral)))


def longitudinal_error(pred: ArrayLike, ref: ArrayLike) -> float:
    """Return mean absolute longitudinal deviation along a reference trajectory."""
    pred_arr = as_trajectory(pred, name="pred")
    ref_arr = as_trajectory(ref, name="ref")
    require_same_shape(pred_arr, ref_arr, "pred", "ref")
    tangents = _reference_tangents(ref_arr)
    deltas = xy(pred_arr) - xy(ref_arr)
    longitudinal = np.sum(deltas * tangents, axis=1)
    return float(np.mean(np.abs(longitudinal)))


def _reference_tangents(ref: FloatArray) -> FloatArray:
    points = xy(ref)
    if points.shape[0] < 2:
        return np.zeros_like(points, dtype=np.float64)

    tangents = np.zeros_like(points, dtype=np.float64)
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    if points.shape[0] > 2:
        tangents[1:-1] = points[2:] - points[:-2]

    norms = vector_norms(tangents)
    nonzero = norms > 1e-12
    tangents[nonzero] = tangents[nonzero] / norms[nonzero, None]
    return tangents

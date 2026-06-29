"""State, action, and workspace coverage metrics."""

from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robometrics.geometry import FloatArray, as_numeric_array, as_points, validate_positive


def workspace_coverage(points: ArrayLike, cell_size: float = 1.0) -> float:
    """Return the number of occupied grid cells visited by sampled positions."""
    point_arr = as_points(points, name="points")
    cell = validate_positive(float(cell_size), name="cell_size")
    cells = np.floor(point_arr / cell).astype(np.int64)
    return float(np.unique(cells, axis=0).shape[0])


def coverage_score(
    samples: ArrayLike,
    bounds: ArrayLike,
    bins: Union[int, ArrayLike] = 10,
) -> float:
    """Return occupied grid-bin coverage for state or action samples."""
    sample_arr = _as_sample_matrix(samples)
    bounds_arr = _as_bounds(bounds, sample_arr.shape[1])
    bin_counts = _as_bin_counts(bins, sample_arr.shape[1])

    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]
    in_bounds = np.all((sample_arr >= lower) & (sample_arr <= upper), axis=1)
    if not np.any(in_bounds):
        return 0.0

    normalized = (sample_arr[in_bounds] - lower) / (upper - lower)
    raw_indices = np.floor(normalized * bin_counts).astype(np.int64)
    indices = np.clip(raw_indices, 0, bin_counts - 1)
    multipliers = np.ones_like(bin_counts)
    for dimension in range(bin_counts.shape[0] - 2, -1, -1):
        multipliers[dimension] = multipliers[dimension + 1] * bin_counts[dimension + 1]
    occupied_arr = np.sum(indices * multipliers, axis=1)
    unique_occupied = int(np.unique(occupied_arr).shape[0])
    total_bins = int(np.prod(bin_counts))
    return float(unique_occupied / total_bins)


def _as_sample_matrix(samples: ArrayLike) -> FloatArray:
    sample_arr = as_numeric_array(samples, name="samples")
    if sample_arr.ndim != 2:
        raise ValueError("samples must be an NxD array")
    if sample_arr.shape[0] == 0 or sample_arr.shape[1] == 0:
        raise ValueError("samples must have non-empty sample and feature dimensions")
    return sample_arr


def _as_bounds(bounds: ArrayLike, dimensions: int) -> FloatArray:
    bounds_arr = as_numeric_array(bounds, name="bounds")
    if bounds_arr.ndim != 2 or bounds_arr.shape != (dimensions, 2):
        raise ValueError("bounds must be a Dx2 array matching sample dimensionality")
    if np.any(bounds_arr[:, 1] <= bounds_arr[:, 0]):
        raise ValueError("each bounds row must have lower < upper")
    return bounds_arr


def _as_bin_counts(bins: Union[int, ArrayLike], dimensions: int) -> NDArray[np.int64]:
    if np.isscalar(bins):
        scalar = float(np.asarray(bins, dtype=np.float64).item())
        if not np.isfinite(scalar) or not scalar.is_integer() or scalar <= 0:
            raise ValueError("bins must be a positive integer or D positive integers")
        return np.full(dimensions, int(scalar), dtype=np.int64)

    bins_arr = as_numeric_array(bins, name="bins")
    if bins_arr.ndim != 1 or bins_arr.shape[0] != dimensions:
        raise ValueError("bins must be a positive integer or D positive integers")
    if np.any(bins_arr <= 0.0) or not np.all(np.equal(bins_arr, np.floor(bins_arr))):
        raise ValueError("bins must be a positive integer or D positive integers")
    return np.asarray(bins_arr, dtype=np.int64)

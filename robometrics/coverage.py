"""Coverage metrics for sampled robot states."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from robometrics.geometry import as_points, validate_positive


def workspace_coverage(points: ArrayLike, cell_size: float = 1.0) -> float:
    """Return the number of occupied grid cells visited by sampled positions."""
    point_arr = as_points(points, name="points")
    cell = validate_positive(float(cell_size), name="cell_size")
    cells = np.floor(point_arr / cell).astype(np.int64)
    return float(np.unique(cells, axis=0).shape[0])

"""Temporal evaluation metrics."""

from __future__ import annotations
from numpy.typing import ArrayLike

from robometrics.geometry import as_1d_array


def compounding_error_index(errors: ArrayLike) -> float:
    """Return final-to-initial error growth for a temporal error sequence."""
    error_arr = as_1d_array(errors, name="errors")
    initial = float(error_arr[0])
    final = float(error_arr[-1])
    if initial == 0.0:
        return 0.0 if final == 0.0 else float("inf")
    return float(final / initial)

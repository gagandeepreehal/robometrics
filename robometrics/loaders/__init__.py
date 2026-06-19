"""Optional dataset loader namespace."""

from robometrics.loaders.generic import load_trajectory_batch, trajectories_to_dataset
from robometrics.loaders.nuscenes import load_nuscenes_trajectories

__all__ = [
    "load_nuscenes_trajectories",
    "load_trajectory_batch",
    "trajectories_to_dataset",
]

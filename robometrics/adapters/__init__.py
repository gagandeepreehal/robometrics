"""Lightweight dataset adapters."""

from __future__ import annotations

from robometrics.adapters.base import TrajectoryAdapter
from robometrics.adapters.generic import GenericCSVAdapter, GenericJSONAdapter
from robometrics.adapters.lerobot import LeRobotStyleAdapter
from robometrics.adapters.mcap import MCAPAdapter
from robometrics.adapters.rlds import RLDSStyleAdapter
from robometrics.adapters.ros_style import ROSStyleAdapter

__all__ = [
    "GenericCSVAdapter",
    "GenericJSONAdapter",
    "LeRobotStyleAdapter",
    "MCAPAdapter",
    "RLDSStyleAdapter",
    "ROSStyleAdapter",
    "TrajectoryAdapter",
    "get_adapter",
]


def get_adapter(name: str) -> TrajectoryAdapter:
    """Return a built-in adapter by short name."""
    normalized = name.strip().lower().replace("_", "-")
    adapters: dict[str, TrajectoryAdapter] = {
        "csv": GenericCSVAdapter(),
        "generic-csv": GenericCSVAdapter(),
        "json": GenericJSONAdapter(),
        "generic-json": GenericJSONAdapter(),
        "ros": ROSStyleAdapter(),
        "ros-style": ROSStyleAdapter(),
        "lerobot": LeRobotStyleAdapter(),
        "lerobot-style": LeRobotStyleAdapter(),
        "rlds": RLDSStyleAdapter(),
        "rlds-style": RLDSStyleAdapter(),
        "mcap": MCAPAdapter(),
    }
    try:
        return adapters[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown adapter: {name}") from exc

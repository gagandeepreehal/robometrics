"""nuScenes-compatible trajectory loading without nuscenes-devkit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from robometrics.io import TrajectoryIOError


def load_nuscenes_trajectories(
    sample_annotation_path: Union[str, Path],
    instance_token: Optional[str] = None,
) -> dict[str, NDArray[np.float64]]:
    """Load agent trajectories from nuScenes sample_annotation.json.

    Format: nuScenes sample_annotation.json is a list of dicts, each with
    fields: token, sample_token, instance_token, translation (list of 3
    floats: x, y, z), rotation, size, etc.

    This function:
      1. Loads the JSON file.
      2. Groups records by instance_token.
      3. For each instance, sorts records by their list position (proxy for
         time, since sample_annotation.json is ordered by scene/sample).
      4. Extracts translation[0:3] as the XYZ position.
      5. Returns {instance_token: Nx3 array}.

    If instance_token is specified, returns only that instance.

    Reference: Caesar et al., nuScenes: A Multimodal Dataset for Autonomous
               Driving, CVPR 2020. Data format:
               https://www.nuscenes.org/data-format

    Inputs:
        sample_annotation_path: path to nuScenes sample_annotation.json
        instance_token: optional filter; if given, only this instance returned.

    Output:
        Dict mapping instance_token to Nx3 numpy array of XYZ positions.

    Raises:
        TrajectoryIOError: if the file does not exist or is not valid JSON
        ValueError: if the JSON does not contain the expected fields
    """
    path = Path(sample_annotation_path)
    if not path.exists():
        raise TrajectoryIOError(f"nuScenes sample_annotation file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TrajectoryIOError(
            f"could not parse nuScenes sample_annotation JSON file {path}: {exc}"
        ) from exc
    except OSError as exc:
        raise TrajectoryIOError(
            f"could not read nuScenes sample_annotation JSON file {path}: {exc}"
        ) from exc

    if not isinstance(payload, list):
        raise ValueError("nuScenes sample_annotation JSON must be a list of records")

    grouped: dict[str, list[list[float]]] = {}
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise ValueError(f"nuScenes sample_annotation record {index} must be an object")
        token = record.get("instance_token")
        if not isinstance(token, str) or not token:
            raise ValueError(
                f"nuScenes sample_annotation record {index} must contain instance_token"
            )
        if instance_token is not None and token != instance_token:
            continue
        translation = record.get("translation")
        if not isinstance(translation, list) or len(translation) < 3:
            raise ValueError(
                f"nuScenes sample_annotation record {index} must contain translation"
            )
        try:
            point = [float(value) for value in translation[:3]]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"nuScenes sample_annotation record {index} translation must be numeric"
            ) from exc
        if not np.all(np.isfinite(point)):
            raise ValueError(
                f"nuScenes sample_annotation record {index} translation must be finite"
            )
        grouped.setdefault(token, []).append(point)

    return {
        token: np.asarray(points, dtype=np.float64)
        for token, points in grouped.items()
    }

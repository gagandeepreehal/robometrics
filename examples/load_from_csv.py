from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robometrics import path_length
from robometrics.io import load_trajectory_csv


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "trajectory.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["x", "y"])
            writer.writeheader()
            writer.writerows(
                [
                    {"x": 0.0, "y": 0.0},
                    {"x": 1.0, "y": 0.0},
                    {"x": 2.0, "y": 0.5},
                ]
            )

        trajectory = load_trajectory_csv(csv_path)
        print(f"Loaded shape: {trajectory.shape}")
        print(f"Path length: {path_length(trajectory):.3f} m")


if __name__ == "__main__":
    main()

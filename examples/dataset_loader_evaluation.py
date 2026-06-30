from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robometrics import Evaluator, load_trajectory_dir
from robometrics.loaders import trajectories_to_dataset


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        pred_dir = root / "predictions"
        gt_dir = root / "references"
        pred_dir.mkdir()
        gt_dir.mkdir()

        _write_csv(pred_dir / "sample_000.csv", [[0.0, 0.0], [1.0, 0.0]])
        _write_csv(gt_dir / "sample_000.csv", [[0.0, 0.0], [1.1, 0.0]])
        _write_json(pred_dir / "sample_001.json", [[0.0, 0.0], [1.2, 0.0]])
        _write_json(gt_dir / "sample_001.json", [[0.0, 0.0], [1.0, 0.0]])

        try:
            loaded = load_trajectory_dir(pred_dir)
            predictions, ground_truths = trajectories_to_dataset(
                sorted(pred_dir.iterdir()),
                sorted(gt_dir.iterdir()),
            )
        except ImportError as exc:
            print(exc)
            return

        result = Evaluator().evaluate_dataset(
            predictions=predictions,
            ground_truths=ground_truths,
            metrics=["ade", "fde"],
            thresholds={"ade": 0.25, "fde": 0.5},
        )

        print("Directory files:", ", ".join(sorted(loaded)))
        print(result.to_markdown())
        print("Schema version:", result.schema_version)


def _write_csv(path: Path, points: list[list[float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["x", "y"])
        writer.writeheader()
        writer.writerows({"x": point[0], "y": point[1]} for point in points)


def _write_json(path: Path, points: list[list[float]]) -> None:
    path.write_text(json.dumps({"points": points}), encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_timing_baselines_script_writes_json(tmp_path) -> None:
    output = tmp_path / "timing.json"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "benchmark_timing_baselines.py"),
            "--repeat",
            "1",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "1"
    assert payload["unit"] == "seconds"
    assert payload["repeat"] == 1
    assert {case["name"] for case in payload["cases"]} == {
        "dataset_evaluator",
        "geometry_safety",
        "manipulation_metrics",
    }
    assert all(case["best"] >= 0.0 for case in payload["cases"])

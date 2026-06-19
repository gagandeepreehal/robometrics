from __future__ import annotations

import subprocess
import sys


def test_python_module_help() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "robometrics", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "list-metrics" in completed.stdout


def test_cli_lists_metrics() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "robometrics", "list-metrics", "--category", "trajectory"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "ade" in completed.stdout
    assert "hausdorff_distance" in completed.stdout

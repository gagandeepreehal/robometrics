from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_all_example_scripts_run() -> None:
    for script in sorted((ROOT / "examples").glob("*.py")):
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        assert completed.stdout.strip()

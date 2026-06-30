from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_cli_csv_evaluate_to_report_pipeline(tmp_path: Path) -> None:
    pred = tmp_path / "pred.csv"
    gt = tmp_path / "gt.csv"
    pred.write_text("t,x,y\n0,0,0\n1,1,0\n2,2,0\n", encoding="utf-8")
    gt.write_text("t,x,y\n0,0,0\n1,1.1,0\n2,2.2,0\n", encoding="utf-8")

    payload, html = _run_evaluate_report(
        tmp_path,
        pred,
        gt,
        "ade",
        "fde",
        "--threshold",
        "ade=0.5",
    )

    assert payload["schema_version"] == "1"
    assert [item["name"] for item in payload["results"]] == ["ade", "fde"]
    assert "RoboMetrics Report" in html


def test_cli_json_evaluate_to_report_pipeline(tmp_path: Path) -> None:
    pred = tmp_path / "pred.json"
    gt = tmp_path / "gt.json"
    pred.write_text(json.dumps({"points": [[0.0, 0.0], [1.0, 0.0]]}), encoding="utf-8")
    gt.write_text(
        json.dumps({"trajectory": [{"x": 0.0, "y": 0.0}, {"x": 1.2, "y": 0.0}]}),
        encoding="utf-8",
    )

    payload, html = _run_evaluate_report(tmp_path, pred, gt, "ade")

    assert payload["summary"]["metric_count"] == 1
    assert abs(float(payload["results"][0]["value"]) - 0.1) < 1e-12
    assert "Metric Values" in html


def test_cli_numpy_evaluate_to_report_pipeline(tmp_path: Path) -> None:
    pred = tmp_path / "pred.npy"
    gt = tmp_path / "gt.npy"
    np.save(pred, np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64))
    np.save(gt, np.array([[0.0, 0.0], [1.0, 0.5]], dtype=np.float64))

    payload, html = _run_evaluate_report(tmp_path, pred, gt, "fde")

    assert payload["results"][0]["name"] == "fde"
    assert payload["results"][0]["value"] == 0.5
    assert "fde" in html


def test_cli_threshold_failure_still_writes_report(tmp_path: Path) -> None:
    pred = tmp_path / "pred.csv"
    gt = tmp_path / "gt.csv"
    pred.write_text("x,y\n0,0\n10,0\n", encoding="utf-8")
    gt.write_text("x,y\n0,0\n0,0\n", encoding="utf-8")

    payload, html = _run_evaluate_report(
        tmp_path,
        pred,
        gt,
        "ade",
        "--threshold",
        "ade=1.0",
        expected_evaluate_code=1,
    )

    assert payload["results"][0]["passed"] is False
    assert payload["summary"]["strict_passed"] is False
    assert "FAIL" in html


def test_cli_hausdorff_pipeline_accepts_different_sample_counts(tmp_path: Path) -> None:
    pred = tmp_path / "pred.json"
    gt = tmp_path / "gt.json"
    pred.write_text(json.dumps({"points": [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]}), encoding="utf-8")
    gt.write_text(json.dumps({"points": [[0.0, 0.0], [2.0, 0.0]]}), encoding="utf-8")

    payload, html = _run_evaluate_report(tmp_path, pred, gt, "hausdorff_distance")

    assert payload["results"][0]["name"] == "hausdorff_distance"
    assert payload["results"][0]["metadata"]["category"] == "trajectory"
    assert "hausdorff_distance" in html


def _run_evaluate_report(
    tmp_path: Path,
    pred: Path,
    gt: Path,
    *metric_args: str,
    expected_evaluate_code: int = 0,
) -> tuple[dict[str, object], str]:
    result_path = tmp_path / "result.json"
    report_path = tmp_path / "report.html"
    evaluate = subprocess.run(
        [
            sys.executable,
            "-m",
            "robometrics",
            "evaluate",
            "--pred",
            str(pred),
            "--gt",
            str(gt),
            "--metrics",
            *metric_args,
            "--output",
            str(result_path),
        ],
        capture_output=True,
        text=True,
    )
    assert evaluate.returncode == expected_evaluate_code, evaluate.stderr
    assert result_path.exists()

    report = subprocess.run(
        [
            sys.executable,
            "-m",
            "robometrics",
            "report",
            str(result_path),
            "--output",
            str(report_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "wrote" in report.stdout
    return json.loads(result_path.read_text(encoding="utf-8")), report_path.read_text(
        encoding="utf-8"
    )

from __future__ import annotations

import json

from robometrics import cli
from robometrics.results import EvaluationResult


def test_cli_evaluate_writes_evaluation_result_json(tmp_path, capsys) -> None:
    pred = tmp_path / "pred.json"
    gt = tmp_path / "gt.json"
    output = tmp_path / "result.json"
    pred.write_text(json.dumps({"points": [[0.0, 0.0], [1.0, 0.0]]}), encoding="utf-8")
    gt.write_text(json.dumps({"points": [[0.0, 0.0], [1.2, 0.0]]}), encoding="utf-8")

    assert (
        cli.main(
            [
                "evaluate",
                "--pred",
                str(pred),
                "--gt",
                str(gt),
                "--metrics",
                "ade",
                "fde",
                "--threshold",
                "ade=1.0",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    result = EvaluationResult.from_json(output.read_text(encoding="utf-8"))
    assert [metric.name for metric in result.results] == ["ade", "fde"]
    assert result.results[0].passed is True
    assert result.metadata["command"] == "evaluate"
    assert "timestamp" in result.metadata
    assert f"wrote {output}" in capsys.readouterr().out


def test_cli_evaluate_reports_missing_file(capsys, tmp_path) -> None:
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps({"points": [[0.0, 0.0], [1.0, 0.0]]}), encoding="utf-8")

    code = cli.main(
        [
            "evaluate",
            "--pred",
            str(tmp_path / "missing.json"),
            "--gt",
            str(gt),
            "--metrics",
            "ade",
            "--output",
            str(tmp_path / "result.json"),
        ]
    )

    assert code == 2
    assert "does not exist" in capsys.readouterr().err


def test_cli_evaluate_reports_invalid_metric(capsys, tmp_path) -> None:
    pred = tmp_path / "pred.json"
    gt = tmp_path / "gt.json"
    for path in (pred, gt):
        path.write_text(json.dumps({"points": [[0.0, 0.0], [1.0, 0.0]]}), encoding="utf-8")

    code = cli.main(
        [
            "evaluate",
            "--pred",
            str(pred),
            "--gt",
            str(gt),
            "--metrics",
            "missing_metric",
            "--output",
            str(tmp_path / "result.json"),
        ]
    )

    assert code == 2
    assert "unknown metric" in capsys.readouterr().err


def test_cli_evaluate_reports_invalid_shape(capsys, tmp_path) -> None:
    pred = tmp_path / "pred.json"
    gt = tmp_path / "gt.json"
    pred.write_text(json.dumps({"points": [[0.0, 0.0], [1.0, 0.0]]}), encoding="utf-8")
    gt.write_text(json.dumps({"points": [[0.0, 0.0]]}), encoding="utf-8")

    code = cli.main(
        [
            "evaluate",
            "--pred",
            str(pred),
            "--gt",
            str(gt),
            "--metrics",
            "ade",
            "--output",
            str(tmp_path / "result.json"),
        ]
    )

    assert code == 2
    assert "not compatible" in capsys.readouterr().err


def test_cli_evaluate_allows_unequal_lengths_for_hausdorff(tmp_path) -> None:
    pred = tmp_path / "pred.json"
    gt = tmp_path / "gt.json"
    output = tmp_path / "result.json"
    pred.write_text(
        json.dumps({"points": [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]]}),
        encoding="utf-8",
    )
    gt.write_text(json.dumps({"points": [[0.0, 0.0], [4.0, 0.0]]}), encoding="utf-8")

    code = cli.main(
        [
            "evaluate",
            "--pred",
            str(pred),
            "--gt",
            str(gt),
            "--metrics",
            "hausdorff_distance",
            "--output",
            str(output),
        ]
    )

    assert code == 0
    result = EvaluationResult.from_json(output.read_text(encoding="utf-8"))
    assert [metric.name for metric in result.results] == ["hausdorff_distance"]
    assert result.results[0].value == 2.0


def test_cli_validate_outputs_human_and_json_reports(tmp_path, capsys) -> None:
    path = tmp_path / "bad.csv"
    output = tmp_path / "validation.json"
    path.write_text("t,x,y\n0,0,0\n0,1,nan\n", encoding="utf-8")

    assert cli.main(["validate", str(path), "--output", str(output)]) == 1

    captured = capsys.readouterr()
    assert "FAIL" in captured.out
    payload = json.loads(output.read_text(encoding="utf-8"))
    codes = {issue["code"] for issue in payload["issues"]}
    assert "invalid_numeric_value" in codes
    assert "non_monotonic_timestamps" in codes


def test_cli_report_writes_static_html(tmp_path, capsys) -> None:
    result = EvaluationResult.from_dict(
        {
            "results": [
                {
                    "name": "ade",
                    "value": 0.1,
                    "unit": "meters",
                    "passed": True,
                    "threshold": 1.0,
                    "metadata": {"category": "trajectory"},
                }
            ],
            "metadata": {"run": "fixture"},
        }
    )
    result_path = tmp_path / "result.json"
    output = tmp_path / "report.html"
    result_path.write_text(result.to_json(), encoding="utf-8")

    assert cli.main(["report", str(result_path), "--output", str(output)]) == 0

    html = output.read_text(encoding="utf-8")
    assert "RoboMetrics Report" in html
    assert "ade" in html
    assert "PASS" in html
    assert f"wrote {output}" in capsys.readouterr().out


def test_cli_benchmark_list_and_run(tmp_path, capsys) -> None:
    pred = tmp_path / "pred.json"
    gt = tmp_path / "gt.json"
    output = tmp_path / "benchmark.json"
    for path in (pred, gt):
        path.write_text(json.dumps({"points": [[0.0, 0.0], [1.0, 0.0]]}), encoding="utf-8")

    assert cli.main(["benchmark", "list"]) == 0
    assert "trajectory_prediction_basic" in capsys.readouterr().out

    assert (
        cli.main(
            [
                "benchmark",
                "run",
                "policy_regression_ci",
                "--pred",
                str(pred),
                "--gt",
                str(gt),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["metadata"]["benchmark_profile"]["name"] == "policy_regression_ci"

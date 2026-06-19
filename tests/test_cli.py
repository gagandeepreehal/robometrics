from __future__ import annotations

import json
import runpy
import subprocess
import sys

from robometrics import cli
from robometrics.results import EvaluationResult, MetricResult


def test_cli_default_help(capsys) -> None:
    assert cli.main([]) == 0

    captured = capsys.readouterr()

    assert "list-metrics" in captured.out
    assert "describe" in captured.out


def test_cli_lists_metrics_directly(capsys) -> None:
    assert cli.main(["list-metrics", "--category", "trajectory"]) == 0

    captured = capsys.readouterr()

    assert "ade" in captured.out
    assert "hausdorff_distance" in captured.out


def test_cli_lists_metrics_as_json(capsys) -> None:
    assert cli.main(["list-metrics", "--category", "trajectory", "--format", "json"]) == 0

    payload = json.loads(capsys.readouterr().out)

    ade = next(metric for metric in payload if metric["name"] == "ade")
    assert ade["reference"] == "Alahi et al., Social Force, CVPR 2016"
    assert ade["is_novel"] is False
    assert "default_kwargs" in ade
    assert "fn" in ade


def test_cli_list_metrics_category_shows_only_requested_category(capsys) -> None:
    assert cli.main(["list-metrics", "--category", "trajectory"]) == 0

    captured = capsys.readouterr()

    assert "| Name | Category | Unit | Reference | Novel |" in captured.out
    assert "ade" in captured.out
    assert "hausdorff_distance" in captured.out
    assert "smoothness_score" not in captured.out


def test_cli_list_metrics_novel_flag_shows_only_novel_metrics(capsys) -> None:
    assert cli.main(["list-metrics", "--novel"]) == 0

    captured = capsys.readouterr()

    assert "jerk_cost" in captured.out
    assert "smoothness_score" in captured.out
    assert "ade" not in captured.out


def test_cli_handles_empty_metric_listing(capsys) -> None:
    assert cli._list_metrics(category="missing", output_format="table") == 0

    assert capsys.readouterr().out == ""


def test_cli_describes_metric(capsys) -> None:
    assert cli.main(["describe", "ade"]) == 0

    captured = capsys.readouterr()

    assert "Reference:   Alahi" in captured.out
    assert "Required inputs: pred, gt" in captured.out


def test_cli_describes_metric_as_json(capsys) -> None:
    assert cli.main(["describe", "average_displacement_error", "--format", "json"]) == 0

    payload = json.loads(capsys.readouterr().out)

    assert payload["name"] == "ade"
    assert payload["aliases"] == ["average_displacement_error"]


def test_cli_prints_version(capsys) -> None:
    assert cli.main(["version"]) == 0

    assert capsys.readouterr().out.strip()


def test_python_module_executes(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["robometrics", "version"])

    try:
        runpy.run_module("robometrics.__main__", run_name="__main__")
    except SystemExit as exc:
        assert exc.code == 0

    assert capsys.readouterr().out.strip()


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


def test_cli_compare_prints_output_and_exits_zero_when_b_wins_thresholded_metric(
    tmp_path,
    capsys,
) -> None:
    result_a = EvaluationResult(
        results=[MetricResult(name="ade", value=2.0, threshold=3.0, passed=True)]
    )
    result_b = EvaluationResult(
        results=[MetricResult(name="ade", value=1.0, threshold=3.0, passed=True)]
    )
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    a_path.write_text(result_a.to_json(), encoding="utf-8")
    b_path.write_text(result_b.to_json(), encoding="utf-8")

    assert cli.main(["compare", str(a_path), str(b_path), "--format", "text"]) == 0

    assert "ade" in capsys.readouterr().out


def test_cli_history_builds_table_from_eval_files(tmp_path, capsys) -> None:
    for step, value in [(0, 3.0), (50, 2.0), (100, 1.0)]:
        result = EvaluationResult(results=[MetricResult(name="ade", value=value)])
        path = tmp_path / f"checkpoint_{step}.eval.json"
        path.write_text(result.to_json(), encoding="utf-8")

    assert cli.main(["history", str(tmp_path)]) == 0

    output = capsys.readouterr().out
    assert "| Step | Label | ade |" in output
    assert "| 0 | checkpoint_0.eval |" in output
    assert "| 50 | checkpoint_50.eval |" in output
    assert "| 100 | checkpoint_100.eval |" in output

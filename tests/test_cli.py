from __future__ import annotations

import json
import runpy
import subprocess
import sys

from robometrics import cli


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

    assert any(metric["name"] == "ade" for metric in payload)


def test_cli_lists_metrics_as_csv(capsys) -> None:
    assert cli.main(["list-metrics", "--category", "trajectory", "--format", "csv"]) == 0

    captured = capsys.readouterr()

    assert captured.out.splitlines()[0] == "name,category,unit,required_inputs,aliases,description"
    assert "ade,trajectory,meters,pred gt,average_displacement_error" in captured.out


def test_cli_handles_empty_metric_listing(capsys) -> None:
    assert cli._list_metrics(category="missing", output_format="text") == 0

    assert capsys.readouterr().out == ""


def test_cli_describes_metric(capsys) -> None:
    assert cli.main(["describe", "ade"]) == 0

    captured = capsys.readouterr()

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

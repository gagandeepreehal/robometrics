"""Command-line entry point for lightweight metric discovery."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TypedDict

from robometrics._version import __version__
from robometrics.benchmarks import list_profiles, run_profile
from robometrics.evaluator import EvaluationInputError, Evaluator
from robometrics.history import EvaluationHistory
from robometrics.io import load_trajectory
from robometrics.registry import MetricDefinition, registry
from robometrics.reporting import write_html_report
from robometrics.results import ComparisonResult, EvaluationResult
from robometrics.validation import validate_dataset


class MetricPayload(TypedDict):
    name: str
    fn: str
    category: str
    unit: str
    required_inputs: list[str]
    default_kwargs: dict[str, Any]
    aliases: list[str]
    reference: str
    is_novel: bool
    higher_is_better: bool
    compatibility: bool
    description: str


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the ``robometrics`` command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "list-metrics":
        return _list_metrics(
            category=args.category,
            output_format=args.format,
            novel_only=args.novel,
        )
    if args.command == "describe":
        return _describe_metric(args.metric, output_format=args.format)
    if args.command == "compare":
        return _compare_results(args.a, args.b, output_format=args.format)
    if args.command == "history":
        return _history_command(args.directory, metric=args.metric, output_format=args.format)
    if args.command == "evaluate":
        return _evaluate_command(
            pred_path=args.pred,
            gt_path=args.gt,
            metrics=args.metrics,
            output_path=args.output,
            threshold_values=args.threshold,
        )
    if args.command == "validate":
        return _validate_command(args.path, output_path=args.output)
    if args.command == "report":
        return _report_command(args.result, output_path=args.output)
    if args.command == "benchmark":
        return _benchmark_command(args)
    if args.command == "version":
        print(__version__)
        return 0
    parser.print_help()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robometrics",
        description="Inspect installed RoboMetrics metrics.",
    )
    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser("list-metrics", help="list registered metrics")
    list_parser.add_argument(
        "--category",
        choices=registry.categories(),
        help="show only metrics in one category",
    )
    list_parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="output format",
    )
    list_parser.add_argument(
        "--novel",
        action="store_true",
        help="show only metrics marked as novel",
    )

    describe_parser = subparsers.add_parser("describe", help="describe one metric")
    describe_parser.add_argument("metric", help="metric name or alias")
    describe_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format",
    )

    compare_parser = subparsers.add_parser("compare", help="compare two evaluation result files")
    compare_parser.add_argument("a", help="first EvaluationResult JSON file")
    compare_parser.add_argument("b", help="second EvaluationResult JSON file")
    compare_parser.add_argument(
        "--format",
        choices=("markdown", "json", "text"),
        default="text",
        help="output format",
    )

    history_parser = subparsers.add_parser(
        "history",
        help="summarize *_eval.json or *.eval.json evaluation history files",
    )
    history_parser.add_argument(
        "directory",
        help="directory containing *_eval.json or *.eval.json files",
    )
    history_parser.add_argument("--metric", help="print values and trend for one metric")
    history_parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="output format",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="evaluate trajectory CSV/JSON predictions against ground truth",
    )
    evaluate_parser.add_argument("--pred", required=True, help="prediction trajectory CSV/JSON")
    evaluate_parser.add_argument("--gt", required=True, help="ground-truth trajectory CSV/JSON")
    evaluate_parser.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help="metric names, for example: ade fde",
    )
    evaluate_parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help="optional pass/fail threshold; repeat for multiple metrics",
    )
    evaluate_parser.add_argument("--output", required=True, help="output EvaluationResult JSON")

    validate_parser = subparsers.add_parser(
        "validate",
        help="validate trajectory-style CSV/JSON data",
    )
    validate_parser.add_argument("path", help="dataset file to validate")
    validate_parser.add_argument("--output", help="optional JSON validation report path")

    report_parser = subparsers.add_parser(
        "report",
        help="generate a static HTML report from EvaluationResult JSON",
    )
    report_parser.add_argument("result", help="EvaluationResult JSON path")
    report_parser.add_argument("--output", required=True, help="output HTML report path")

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="list or run built-in benchmark profiles",
    )
    benchmark_subparsers = benchmark_parser.add_subparsers(dest="benchmark_command")
    benchmark_subparsers.add_parser("list", help="list benchmark profiles")
    benchmark_run = benchmark_subparsers.add_parser("run", help="run a benchmark profile")
    benchmark_run.add_argument("profile", help="profile name")
    benchmark_run.add_argument("--pred", required=True, help="prediction trajectory CSV/JSON")
    benchmark_run.add_argument("--gt", required=True, help="ground-truth trajectory CSV/JSON")
    benchmark_run.add_argument("--output", required=True, help="output EvaluationResult JSON")

    subparsers.add_parser("version", help="print the installed package version")
    return parser


def _list_metrics(*, category: Optional[str], output_format: str, novel_only: bool = False) -> int:
    metrics = registry.list_metrics(category=category)
    if novel_only:
        metrics = [metric for metric in metrics if metric.is_novel]
    if not metrics:
        return 0
    rows = [_metric_payload(metric) for metric in metrics]
    if output_format == "json":
        print(json.dumps(rows, sort_keys=True))
        return 0

    print("| Name | Category | Unit | Direction | Reference | Novel |")
    print("| --- | --- | --- | --- | --- | --- |")
    for row in rows:
        print(
            "| "
            + " | ".join(
                [
                    row["name"],
                    row["category"],
                    row["unit"] or "-",
                    "higher" if row["higher_is_better"] else "lower",
                    row["reference"] or "-",
                    str(row["is_novel"]),
                ]
            )
            + " |"
        )
    return 0


def _describe_metric(metric_name: str, *, output_format: str) -> int:
    metric = registry.get(metric_name)
    payload = _metric_payload(metric)
    if output_format == "json":
        print(json.dumps(payload, sort_keys=True))
        return 0
    print(f"Name:        {payload['name']}")
    print(f"Category:    {payload['category']}")
    print(f"Unit:        {payload['unit'] or '-'}")
    print(f"Reference:   {payload['reference'] or '-'}")
    print(f"Novel:       {payload['is_novel']}")
    direction = "higher is better" if payload["higher_is_better"] else "lower is better"
    print(f"Direction:   {direction}")
    print(f"Description: {payload['description'] or '-'}")
    print(f"Aliases:     {', '.join(payload['aliases']) or '-'}")
    print(f"Required inputs: {', '.join(payload['required_inputs']) or '-'}")
    return 0


def _compare_results(a_path: str, b_path: str, *, output_format: str) -> int:
    result_a = EvaluationResult.from_json(Path(a_path).read_text(encoding="utf-8"))
    result_b = EvaluationResult.from_json(Path(b_path).read_text(encoding="utf-8"))
    comparison = result_a.compare(result_b)

    if output_format == "json":
        print(comparison.to_json())
        return _compare_exit_code(result_a, result_b, comparison)
    if output_format == "markdown":
        print(comparison.to_markdown())
        return _compare_exit_code(result_a, result_b, comparison)

    for item in comparison.comparisons:
        print(
            f"{item.name}: A={item.value_a:.6g} B={item.value_b:.6g} "
            f"Delta={item.delta:.6g} Winner={item.winner}"
        )
    return _compare_exit_code(result_a, result_b, comparison)


def _history_command(directory: str, *, metric: Optional[str], output_format: str) -> int:
    history = _load_history(Path(directory))
    if output_format == "json":
        if metric is None:
            print(history.to_json())
        else:
            values = history.metric_values(metric)
            print(
                json.dumps(
                    {
                        "metric": metric,
                        "values": {
                            str(step): _json_number(value)
                            for step, value in values.items()
                        },
                        "trend": _json_number(history.trend(metric)),
                    },
                    allow_nan=False,
                    sort_keys=True,
                )
            )
        return 0

    if metric is None:
        print(history.to_markdown())
        return 0

    values = history.metric_values(metric)
    print(f"| Step | {metric} |")
    print("| --- | ---: |")
    for step, value in values.items():
        print(f"| {step} | {_format_value(value)} |")
    print(f"Trend: {_format_value(history.trend(metric))}")
    return 0


def _evaluate_command(
    *,
    pred_path: str,
    gt_path: str,
    metrics: Sequence[str],
    output_path: str,
    threshold_values: Sequence[str],
) -> int:
    try:
        pred = _load_cli_trajectory(pred_path, label="--pred")
        gt = _load_cli_trajectory(gt_path, label="--gt")
        thresholds = _parse_thresholds(threshold_values)
        result = Evaluator().evaluate(
            prediction=pred,
            ground_truth=gt,
            metrics=list(metrics),
            thresholds=thresholds,
        )
        _require_cli_metric_success(result)
    except Exception as exc:  # noqa: BLE001 - CLI errors must be printed cleanly.
        print(f"robometrics evaluate: {exc}", file=sys.stderr)
        return 2

    result.metadata.update(
        {
            "command": "evaluate",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "inputs": {"pred": str(pred_path), "gt": str(gt_path)},
            "metric_names": list(metrics),
        }
    )
    Path(output_path).write_text(result.to_json(), encoding="utf-8")
    print(f"wrote {output_path}")
    return 0 if result.strict_passed is not False else 1


def _validate_command(path: str, *, output_path: Optional[str]) -> int:
    result = validate_dataset(path)
    print(result.to_text())
    if output_path is not None:
        Path(output_path).write_text(result.to_json(), encoding="utf-8")
    return 0 if result.passed else 1


def _report_command(result_path: str, *, output_path: str) -> int:
    try:
        output = write_html_report(result_path, output_path)
    except Exception as exc:  # noqa: BLE001 - CLI errors must be printed cleanly.
        print(f"robometrics report: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {output}")
    return 0


def _benchmark_command(args: argparse.Namespace) -> int:
    if args.benchmark_command == "list":
        print("| Name | Metrics | Description |")
        print("| --- | --- | --- |")
        for profile in list_profiles():
            print(
                "| "
                + " | ".join(
                    [
                        profile.name,
                        ", ".join(profile.metrics),
                        profile.description,
                    ]
                )
                + " |"
            )
        return 0

    if args.benchmark_command == "run":
        try:
            pred = _load_cli_trajectory(args.pred, label="--pred")
            gt = _load_cli_trajectory(args.gt, label="--gt")
            _require_matching_cli_shapes(pred, gt)
            result = run_profile(args.profile, prediction=pred, ground_truth=gt)
        except Exception as exc:  # noqa: BLE001 - CLI errors must be printed cleanly.
            print(f"robometrics benchmark run: {exc}", file=sys.stderr)
            return 2
        result.metadata.update(
            {
                "command": "benchmark run",
                "inputs": {"pred": str(args.pred), "gt": str(args.gt)},
            }
        )
        Path(args.output).write_text(result.to_json(), encoding="utf-8")
        print(f"wrote {args.output}")
        return 0 if result.strict_passed is not False else 1

    print("robometrics benchmark: expected 'list' or 'run'", file=sys.stderr)
    return 2


def _load_cli_trajectory(path: str, *, label: str) -> Any:
    trajectory_path = Path(path)
    if not trajectory_path.exists():
        raise EvaluationInputError(f"{label} file does not exist: {trajectory_path}")
    if not trajectory_path.is_file():
        raise EvaluationInputError(f"{label} path is not a file: {trajectory_path}")
    try:
        return load_trajectory(trajectory_path)
    except Exception as exc:  # noqa: BLE001 - preserve loader context in CLI message.
        raise EvaluationInputError(f"could not load {label} {trajectory_path}: {exc}") from exc


def _require_matching_cli_shapes(pred: Any, gt: Any) -> None:
    pred_shape = getattr(pred, "shape", None)
    gt_shape = getattr(gt, "shape", None)
    if pred_shape != gt_shape:
        raise EvaluationInputError(
            f"--pred and --gt must have the same shape; got {pred_shape} and {gt_shape}"
        )


def _require_cli_metric_success(result: EvaluationResult) -> None:
    if result.results and all("error" in metric.metadata for metric in result.results):
        message = str(result.results[0].metadata.get("error") or "no metric could be evaluated")
        raise EvaluationInputError(message)


def _parse_thresholds(values: Sequence[str]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for raw in values:
        if "=" not in raw:
            raise EvaluationInputError("thresholds must use METRIC=VALUE syntax")
        name, value = raw.split("=", 1)
        metric = registry.get(name)
        try:
            threshold = float(value)
        except ValueError as exc:
            raise EvaluationInputError(f"threshold for {name} must be numeric") from exc
        if not math.isfinite(threshold):
            raise EvaluationInputError(f"threshold for {name} must be finite")
        thresholds[metric.name] = threshold
    return thresholds


def _metric_payload(metric: MetricDefinition) -> MetricPayload:
    return {
        "name": metric.name,
        "fn": _callable_name(metric.fn),
        "category": metric.category,
        "unit": metric.unit,
        "required_inputs": list(metric.required_inputs),
        "default_kwargs": dict(metric.default_kwargs),
        "aliases": list(metric.aliases),
        "reference": metric.reference,
        "is_novel": metric.is_novel,
        "higher_is_better": metric.higher_is_better,
        "compatibility": metric.compatibility is not None,
        "description": metric.description,
    }


def _compare_exit_code(
    result_a: EvaluationResult,
    result_b: EvaluationResult,
    comparison: ComparisonResult,
) -> int:
    thresholded_names = {
        metric.name
        for metric in [*result_a.results, *result_b.results]
        if metric.threshold is not None
    }
    if not thresholded_names:
        return 0
    winners = {
        item.name: item.winner
        for item in comparison.comparisons
        if item.name in thresholded_names
    }
    comparisons = {
        item.name: item
        for item in comparison.comparisons
        if item.name in thresholded_names
    }
    for name, winner in winners.items():
        item = comparisons[name]
        if winner == "b":
            continue
        if (
            winner == "tie"
            and math.isfinite(item.value_a)
            and math.isfinite(item.value_b)
            and math.isclose(item.value_a, item.value_b, rel_tol=1e-12, abs_tol=1e-12)
        ):
            continue
        return 1
    return 0


def _load_history(directory: Path) -> EvaluationHistory:
    files = sorted({*directory.glob("*_eval.json"), *directory.glob("*.eval.json")})
    history = EvaluationHistory()
    for fallback_step, path in enumerate(files):
        result = EvaluationResult.from_json(path.read_text(encoding="utf-8"))
        history.record(
            step=_infer_step(path, fallback_step),
            result=result,
            label=path.stem,
        )
    return history


def _infer_step(path: Path, fallback_step: int) -> int:
    matches = re.findall(r"\d+", path.stem)
    if not matches:
        return fallback_step
    return int(matches[-1])


def _callable_name(value: object) -> str:
    module = getattr(value, "__module__", "")
    name = getattr(value, "__name__", repr(value))
    return f"{module}.{name}" if module else str(name)


def _format_value(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value) and value > 0.0:
        return "inf"
    if math.isinf(value) and value < 0.0:
        return "-inf"
    return f"{value:.6g}"


def _json_number(value: float) -> Optional[float]:
    return float(value) if math.isfinite(value) else None


if __name__ == "__main__":
    raise SystemExit(main())

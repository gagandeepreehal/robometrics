"""Command-line entry point for lightweight metric discovery."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, TypedDict

from robometrics._version import __version__
from robometrics.history import EvaluationHistory
from robometrics.registry import MetricDefinition, registry
from robometrics.results import ComparisonResult, EvaluationResult


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

    history_parser = subparsers.add_parser("history", help="summarize evaluation history files")
    history_parser.add_argument("directory", help="directory containing *_eval.json files")
    history_parser.add_argument("--metric", help="print values and trend for one metric")
    history_parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="output format",
    )

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

    print("| Name | Category | Unit | Reference | Novel |")
    print("| --- | --- | --- | --- | --- |")
    for row in rows:
        print(
            "| "
            + " | ".join(
                [
                    row["name"],
                    row["category"],
                    row["unit"] or "-",
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
    return 0 if all(winner == "b" for winner in winners.values()) else 1


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
    if value != value:
        return "nan"
    if value == float("inf"):
        return "inf"
    if value == float("-inf"):
        return "-inf"
    return f"{value:.6g}"


def _json_number(value: float) -> Optional[float]:
    return float(value) if value == value and value not in {float("inf"), float("-inf")} else None


if __name__ == "__main__":
    raise SystemExit(main())

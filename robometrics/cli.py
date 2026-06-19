"""Command-line entry point for lightweight metric discovery."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Optional, TypedDict

from robometrics._version import __version__
from robometrics.registry import MetricDefinition, registry


class MetricPayload(TypedDict):
    name: str
    category: str
    unit: str
    required_inputs: list[str]
    aliases: list[str]
    reference: str
    is_novel: bool
    description: str


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the ``robometrics`` command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "list-metrics":
        return _list_metrics(category=args.category, output_format=args.format)
    if args.command == "describe":
        return _describe_metric(args.metric, output_format=args.format)
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
        choices=("text", "json", "csv"),
        default="text",
        help="output format",
    )

    describe_parser = subparsers.add_parser("describe", help="describe one metric")
    describe_parser.add_argument("metric", help="metric name or alias")
    describe_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format",
    )

    subparsers.add_parser("version", help="print the installed package version")
    return parser


def _list_metrics(*, category: Optional[str], output_format: str) -> int:
    metrics = registry.list_metrics(category=category)
    if not metrics:
        return 0
    rows = [_metric_payload(metric) for metric in metrics]
    if output_format == "json":
        print(json.dumps(rows, sort_keys=True))
        return 0
    if output_format == "csv":
        print("name,category,unit,required_inputs,aliases,description")
        for row in rows:
            print(
                ",".join(
                    [
                        row["name"],
                        row["category"],
                        row["unit"],
                        " ".join(row["required_inputs"]),
                        " ".join(row["aliases"]),
                        row["description"],
                    ]
                )
            )
        return 0

    name_width = max(len(metric.name) for metric in metrics)
    category_width = max(len(metric.category) for metric in metrics)
    for metric in metrics:
        print(
            f"{metric.name:<{name_width}}  "
            f"{metric.category:<{category_width}}  "
            f"{metric.unit or '-'}"
        )
    return 0


def _describe_metric(metric_name: str, *, output_format: str) -> int:
    metric = registry.get(metric_name)
    payload = _metric_payload(metric)
    if output_format == "json":
        print(json.dumps(payload, sort_keys=True))
        return 0
    print(f"Name: {payload['name']}")
    print(f"Category: {payload['category']}")
    print(f"Unit: {payload['unit'] or '-'}")
    print(f"Required inputs: {', '.join(payload['required_inputs']) or '-'}")
    print(f"Aliases: {', '.join(payload['aliases']) or '-'}")
    print(f"Reference: {payload['reference'] or '-'}")
    print(f"Novel: {payload['is_novel']}")
    print(f"Description: {payload['description'] or '-'}")
    return 0


def _metric_payload(metric: MetricDefinition) -> MetricPayload:
    return {
        "name": metric.name,
        "category": metric.category,
        "unit": metric.unit,
        "required_inputs": list(metric.required_inputs),
        "aliases": list(metric.aliases),
        "reference": metric.reference,
        "is_novel": metric.is_novel,
        "description": metric.description,
    }


if __name__ == "__main__":
    raise SystemExit(main())

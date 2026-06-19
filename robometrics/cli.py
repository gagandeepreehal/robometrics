"""Command-line entry point for lightweight metric discovery."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Optional

from robometrics._version import __version__
from robometrics.registry import registry


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the ``robometrics`` command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "list-metrics":
        return _list_metrics(category=args.category)
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

    subparsers.add_parser("version", help="print the installed package version")
    return parser


def _list_metrics(*, category: Optional[str]) -> int:
    metrics = registry.list_metrics(category=category)
    if not metrics:
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


if __name__ == "__main__":
    raise SystemExit(main())

"""Static HTML report generation for evaluation results."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Optional, Union

from robometrics.results import EvaluationResult

PathLike = Union[str, Path]


def write_html_report(result_path: PathLike, output_path: PathLike) -> Path:
    """Read an EvaluationResult JSON file and write a lightweight HTML report."""
    source = Path(result_path)
    output = Path(output_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("result JSON must contain an object")
    result = EvaluationResult.from_dict(payload)
    output.write_text(render_html_report(result, raw_payload=payload), encoding="utf-8")
    return output


def render_html_report(
    result: EvaluationResult,
    *,
    raw_payload: Optional[dict[str, Any]] = None,
) -> str:
    """Return a complete static HTML report for an evaluation result."""
    raw = raw_payload or result.to_dict()
    summary = result.summary()
    metadata = result.metadata
    comparison = _comparison_payload(raw)
    schema_version = raw.get("schema_version", result.schema_version)
    rows = "\n".join(_metric_row(metric.to_dict()) for metric in result.results)

    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "  <title>RoboMetrics Report</title>\n"
        "  <style>\n"
        "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; "
        "margin: 2rem; color: #17202a; }\n"
        "    table { border-collapse: collapse; width: 100%; margin: 1rem 0; }\n"
        "    th, td { border: 1px solid #d5dde5; padding: .5rem .6rem; text-align: left; }\n"
        "    th { background: #f3f6f8; }\n"
        "    .pass { color: #0b6b3a; font-weight: 700; }\n"
        "    .fail { color: #a82121; font-weight: 700; }\n"
        "    .unknown { color: #5b6470; }\n"
        "    code, pre { background: #f6f8fa; border-radius: 4px; }\n"
        "    pre { padding: 1rem; overflow-x: auto; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        "  <h1>RoboMetrics Report</h1>\n"
        "  <h2>Summary</h2>\n"
        f"  {_summary_table(summary, schema_version=schema_version)}\n"
        "  <h2>Metric Values</h2>\n"
        "  <table>\n"
        "    <thead><tr><th>Metric</th><th>Value</th><th>Unit</th>"
        "<th>Passed</th><th>Threshold</th></tr></thead>\n"
        f"    <tbody>\n{rows}\n    </tbody>\n"
        "  </table>\n"
        "  <h2>Metadata</h2>\n"
        f"  <pre>{_json_block(metadata)}</pre>\n"
        f"  {_comparison_section(comparison)}\n"
        "</body>\n"
        "</html>\n"
    )


def _metric_row(metric: dict[str, Any]) -> str:
    passed = metric.get("passed")
    klass = "unknown"
    label = "-"
    if passed is True:
        klass = "pass"
        label = "PASS"
    elif passed is False:
        klass = "fail"
        label = "FAIL"
    threshold = metric.get("threshold")
    return (
        "      <tr>"
        f"<td>{_escape(metric.get('name'))}</td>"
        f"<td>{_escape(metric.get('value'))}</td>"
        f"<td>{_escape(metric.get('unit') or '-')}</td>"
        f'<td class="{klass}">{label}</td>'
        f"<td>{_escape('-' if threshold is None else threshold)}</td>"
        "</tr>"
    )


def _summary_table(summary: dict[str, Any], *, schema_version: Any) -> str:
    rows = []
    rows.append(
        f"<tr><th>{_escape('schema_version')}</th><td>{_escape(schema_version)}</td></tr>"
    )
    keys = (
        "metric_count",
        "passed",
        "strict_passed",
        "passed_count",
        "failed_count",
        "error_count",
    )
    for key in keys:
        rows.append(f"<tr><th>{_escape(key)}</th><td>{_escape(summary.get(key))}</td></tr>")
    return "<table><tbody>" + "".join(rows) + "</tbody></table>"


def _comparison_payload(raw: dict[str, Any]) -> Any:
    if "baseline_comparison" in raw:
        return raw["baseline_comparison"]
    metadata = raw.get("metadata")
    if isinstance(metadata, dict):
        return metadata.get("baseline_comparison")
    return None


def _comparison_section(comparison: Any) -> str:
    if comparison is None:
        return ""
    return "  <h2>Baseline Comparison</h2>\n" f"  <pre>{_json_block(comparison)}</pre>\n"


def _json_block(value: Any) -> str:
    return html.escape(json.dumps(value, allow_nan=False, indent=2, sort_keys=True))


def _escape(value: Any) -> str:
    return html.escape(str(value))

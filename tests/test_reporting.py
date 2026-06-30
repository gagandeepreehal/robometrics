from __future__ import annotations

import json

import pytest

from robometrics.reporting import render_html_report, write_html_report
from robometrics.results import EvaluationResult, MetricResult


def test_write_html_report_rejects_non_object_json(tmp_path) -> None:
    result_path = tmp_path / "result.json"
    output_path = tmp_path / "report.html"
    result_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="result JSON must contain an object"):
        write_html_report(result_path, output_path)

    assert not output_path.exists()


def test_render_html_report_escapes_values_and_renders_statuses() -> None:
    result = EvaluationResult(
        results=[
            MetricResult(
                name="<bad>",
                value=2.0,
                unit=None,
                passed=False,
                threshold=None,
            ),
            MetricResult(name="unchecked", value=1.0, passed=None),
        ],
        metadata={"note": "<script>"},
    )

    html = render_html_report(result)

    assert "schema_version" in html
    assert "<td>1</td>" in html
    assert "&lt;bad&gt;" in html
    assert "&lt;script&gt;" in html
    assert '<td class="fail">FAIL</td>' in html
    assert '<td class="unknown">-</td>' in html
    assert "<td>-</td>" in html


def test_render_html_report_includes_root_baseline_comparison() -> None:
    result = EvaluationResult(results=[MetricResult(name="ade", value=1.0)])

    html = render_html_report(
        result,
        raw_payload={
            "results": [metric.to_dict() for metric in result.results],
            "baseline_comparison": {"winner": "candidate"},
        },
    )

    assert "Baseline Comparison" in html
    assert "candidate" in html


def test_render_html_report_includes_metadata_baseline_comparison() -> None:
    result = EvaluationResult(
        results=[MetricResult(name="ade", value=1.0)],
        metadata={"baseline_comparison": {"winner": "baseline"}},
    )

    html = render_html_report(result)

    assert "Baseline Comparison" in html
    assert "baseline" in html


def test_render_html_report_skips_comparison_when_metadata_is_not_an_object() -> None:
    result = EvaluationResult(results=[MetricResult(name="ade", value=1.0)])

    html = render_html_report(
        result,
        raw_payload={
            "results": [metric.to_dict() for metric in result.results],
            "metadata": [],
        },
    )

    assert "Baseline Comparison" not in html


def test_write_html_report_round_trips_baseline_comparison_payload(tmp_path) -> None:
    result_path = tmp_path / "result.json"
    output_path = tmp_path / "report.html"
    result_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "results": [{"name": "ade", "value": 1.0}],
                "metadata": {"run": "fixture"},
                "baseline_comparison": {"winner": "candidate"},
            }
        ),
        encoding="utf-8",
    )

    returned = write_html_report(result_path, output_path)

    assert returned == output_path
    html = output_path.read_text(encoding="utf-8")
    assert "schema_version" in html
    assert "Baseline Comparison" in html
    assert "candidate" in html

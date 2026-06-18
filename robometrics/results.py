"""Structured metric and evaluation results."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class MetricResult(BaseModel):
    """Result for one metric execution."""

    name: str
    value: float
    unit: str = ""
    passed: bool | None = None
    threshold: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("unit", mode="before")
    @classmethod
    def _normalize_unit(cls, unit: str | None) -> str:
        return "" if unit is None else unit

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "value": _json_safe(self.value),
            "unit": self.unit,
            "passed": self.passed,
            "threshold": _json_safe(self.threshold),
            "metadata": _json_safe(self.metadata),
        }

    def to_json(self) -> str:
        """Return a JSON string representation."""
        return json.dumps(self.to_dict(), allow_nan=False, sort_keys=True)


class EvaluationResult(BaseModel):
    """Collection of metric results from one evaluation run."""

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    results: list[MetricResult] = Field(
        default_factory=list,
        validation_alias=AliasChoices("results", "metrics"),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def metrics(self) -> list[MetricResult]:
        """Backward-compatible alias for ``results``."""
        return self.results

    @metrics.setter
    def metrics(self, value: list[MetricResult]) -> None:
        self.results = value

    @property
    def passed(self) -> bool | None:
        """Return aggregate pass status when all metric results define pass/fail."""
        statuses = [metric.passed for metric in self.results]
        if not statuses or any(status is None for status in statuses):
            return None
        return all(statuses)

    def summary(self) -> dict[str, Any]:
        """Return metric counts, categories, pass status, and aggregate statistics."""
        values = np.asarray(
            [metric.value for metric in self.results if np.isfinite(metric.value)],
            dtype=np.float64,
        )
        categories = sorted(
            {
                str(category)
                for category in (metric.metadata.get("category") for metric in self.results)
                if category
            }
        )
        aggregate: dict[str, float | int | None] = {"count": int(values.size)}
        if values.size:
            aggregate.update(
                {
                    "mean": float(np.mean(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values)),
                }
            )
        else:
            aggregate.update({"mean": None, "min": None, "max": None, "median": None, "std": None})

        return {
            "metric_count": len(self.results),
            "categories": categories,
            "passed": self.passed,
            "passed_count": sum(metric.passed is True for metric in self.results),
            "failed_count": sum(metric.passed is False for metric in self.results),
            "error_count": sum("error" in metric.metadata for metric in self.results),
            "aggregate": aggregate,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        return {
            "results": [metric.to_dict() for metric in self.results],
            "summary": self.summary(),
            "metadata": _json_safe(self.metadata),
        }

    def to_json(self) -> str:
        """Return a JSON string representation."""
        return json.dumps(self.to_dict(), allow_nan=False, sort_keys=True)

    def to_markdown(self) -> str:
        """Return a GitHub-flavored Markdown table."""
        lines = [
            "| Metric | Value | Unit | Passed | Threshold |",
            "| --- | ---: | --- | --- | ---: |",
        ]
        for metric in self.results:
            threshold = "" if metric.threshold is None else _format_float(metric.threshold)
            passed = "" if metric.passed is None else str(metric.passed)
            lines.append(
                "| "
                + " | ".join(
                    [
                        _display_name(metric.name),
                        _format_float(metric.value),
                        metric.unit,
                        passed,
                        threshold,
                    ]
                )
                + " |"
            )
        return "\n".join(lines)

    def to_dataframe(self) -> Any:
        """Return a pandas DataFrame with one row per metric result."""
        import pandas as pd

        rows = []
        for metric in self.results:
            rows.append(
                {
                    "name": metric.name,
                    "value": metric.value,
                    "unit": metric.unit,
                    "passed": metric.passed,
                    "threshold": metric.threshold,
                    "category": metric.metadata.get("category"),
                    "error": metric.metadata.get("error"),
                }
            )
        return pd.DataFrame(rows)


def _display_name(name: str) -> str:
    special = {
        "ade": "ADE",
        "fde": "FDE",
        "min_ade": "minADE",
        "min_fde": "minFDE",
        "miss_rate": "Miss Rate",
    }
    return special.get(name, name.replace("_", " ").title())


def _format_float(value: float) -> str:
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return f"{value:.6g}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value

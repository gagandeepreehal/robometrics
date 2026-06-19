"""Structured metric and evaluation results."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np


@dataclass
class MetricResult:
    """Result for one metric execution.

    Args:
        name: Stable metric name.
        value: Numeric metric value in ``unit``.
        unit: Unit label, for example ``"m"``, ``"m/s^2"``, or ``""`` for unitless values.
        passed: Optional threshold pass/fail status.
        threshold: Optional threshold used to compute ``passed``.
        metadata: Extra JSON-compatible context such as category or error details.
    """

    name: str
    value: float
    unit: str = ""
    passed: Optional[bool] = None
    threshold: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.value = float(self.value)
        self.unit = "" if self.unit is None else str(self.unit)
        self.threshold = None if self.threshold is None else float(self.threshold)
        self.metadata = dict(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        metadata = dict(self.metadata)
        if not np.isfinite(self.value):
            metadata.setdefault(
                "value_serialization",
                {
                    "original": _non_finite_label(self.value),
                    "json_value": None,
                },
            )
        return {
            "name": self.name,
            "value": _json_safe(self.value),
            "unit": self.unit,
            "passed": self.passed,
            "threshold": _json_safe(self.threshold),
            "metadata": _json_safe(metadata),
        }

    def to_json(self) -> str:
        """Return a JSON string representation."""
        return json.dumps(self.to_dict(), allow_nan=False, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MetricResult:
        """Create a result from a dictionary produced by ``to_dict()``."""
        metadata = dict(payload.get("metadata") or {})
        return cls(
            name=str(payload["name"]),
            value=_restore_json_number(payload.get("value"), metadata),
            unit=str(payload.get("unit") or ""),
            passed=payload.get("passed"),
            threshold=(
                None
                if payload.get("threshold") is None
                else _restore_json_number(payload.get("threshold"), {})
            ),
            metadata=metadata,
        )

    @classmethod
    def from_json(cls, payload: str) -> MetricResult:
        """Create a result from a JSON string produced by ``to_json()``."""
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("MetricResult JSON must decode to an object")
        return cls.from_dict(data)


@dataclass(init=False)
class EvaluationResult:
    """Collection of metric results from one local evaluation run.

    Args:
        results: Metric results to store.
        metrics: Backward-compatible alias for ``results``.
        metadata: Extra JSON-compatible run metadata.
    """

    results: list[MetricResult]
    metadata: dict[str, Any]

    def __init__(
        self,
        results: Optional[list[MetricResult]] = None,
        metadata: Optional[dict[str, Any]] = None,
        *,
        metrics: Optional[list[MetricResult]] = None,
    ) -> None:
        if results is not None and metrics is not None:
            raise ValueError("provide either results or metrics, not both")
        selected = results if results is not None else metrics
        self.results = list(selected or [])
        self.metadata = dict(metadata or {})

    @property
    def metrics(self) -> list[MetricResult]:
        """Backward-compatible alias for ``results``."""
        return self.results

    @metrics.setter
    def metrics(self, value: list[MetricResult]) -> None:
        self.results = value

    @property
    def passed(self) -> Optional[bool]:
        """Return aggregate pass status when all metric results define pass/fail."""
        statuses = [metric.passed for metric in self.results]
        if not statuses or any(status is None for status in statuses):
            return None
        return all(statuses)

    @property
    def strict_passed(self) -> Optional[bool]:
        """Return pass status considering only metrics with thresholds."""
        statuses = [metric.passed for metric in self.results if metric.passed is not None]
        if not statuses:
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
        finite_units = {
            metric.unit
            for metric in self.results
            if np.isfinite(metric.value)
        }
        mixed_units = len(finite_units) > 1
        aggregate: dict[str, Optional[Union[float, int, str]]] = {"count": int(values.size)}
        if values.size and not mixed_units:
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
        if mixed_units:
            aggregate["warning"] = "aggregate mixes metric units; use per_unit summaries"
        aggregate["unit_count"] = len(finite_units)

        summary = {
            "metric_count": len(self.results),
            "categories": categories,
            "passed": self.passed,
            "strict_passed": self.strict_passed,
            "passed_count": sum(metric.passed is True for metric in self.results),
            "failed_count": sum(metric.passed is False for metric in self.results),
            "error_count": sum("error" in metric.metadata for metric in self.results),
            "aggregate": aggregate,
        }
        if mixed_units:
            summary["per_unit"] = _summaries_by_unit(self.results)
        return summary

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

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvaluationResult:
        """Create an evaluation result from a dictionary produced by ``to_dict()``."""
        raw_results = payload.get("results")
        if not isinstance(raw_results, list):
            raise ValueError("EvaluationResult payload must contain a results list")
        metadata = payload.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("EvaluationResult metadata must be an object")
        results = []
        for result in raw_results:
            if not isinstance(result, dict):
                raise ValueError("EvaluationResult results must contain objects")
            results.append(MetricResult.from_dict(result))
        return cls(
            results=results,
            metadata=metadata,
        )

    @classmethod
    def from_json(cls, payload: str) -> EvaluationResult:
        """Create an evaluation result from a JSON string produced by ``to_json()``."""
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("EvaluationResult JSON must decode to an object")
        return cls.from_dict(data)

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

    def to_csv(self, path: Optional[Union[str, Path]] = None) -> str:
        """Return CSV text, optionally writing it to ``path``."""
        csv_text = str(self.to_dataframe().to_csv(index=False))
        if path is not None:
            Path(path).write_text(csv_text, encoding="utf-8")
        return csv_text


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


def _restore_json_number(value: Any, metadata: dict[str, Any]) -> float:
    if value is not None:
        return float(value)
    serialization = metadata.get("value_serialization")
    if isinstance(serialization, dict):
        original = serialization.get("original")
        if original == "nan":
            return float("nan")
        if original == "inf":
            return float("inf")
        if original == "-inf":
            return float("-inf")
    return float("nan")


def _non_finite_label(value: float) -> str:
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return str(value)


def _summaries_by_unit(
    results: list[MetricResult],
) -> dict[str, dict[str, Optional[Union[float, int]]]]:
    summaries: dict[str, dict[str, Optional[Union[float, int]]]] = {}
    units = sorted({metric.unit for metric in results if np.isfinite(metric.value)})
    for unit in units:
        values = np.asarray(
            [
                metric.value
                for metric in results
                if metric.unit == unit and np.isfinite(metric.value)
            ],
            dtype=np.float64,
        )
        label = unit or "unitless"
        summaries[label] = {
            "count": int(values.size),
            "mean": float(np.mean(values)) if values.size else None,
            "min": float(np.min(values)) if values.size else None,
            "max": float(np.max(values)) if values.size else None,
            "median": float(np.median(values)) if values.size else None,
            "std": float(np.std(values)) if values.size else None,
        }
    return summaries

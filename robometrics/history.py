"""Evaluation history and trend tracking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from robometrics.results import EvaluationResult


@dataclass
class CheckpointEntry:
    """One recorded evaluation checkpoint."""

    step: int
    result: EvaluationResult
    label: Optional[str] = None


class EvaluationHistory:
    """Track EvaluationResult objects across training checkpoints.

    Stores results keyed by integer step number. Provides trend analysis
    and comparison across checkpoints for CI integration and experiment
    tracking.

    Usage:
        history = EvaluationHistory()
        for step, result in training_loop():
            history.record(step=step, result=result)
        print(history.trend("ade"))        # slope of ADE over steps
        print(history.best_step("ade"))    # step with lowest ADE
        print(history.to_markdown())       # table of all checkpoints
    """

    def __init__(self) -> None:
        self._entries: list[CheckpointEntry] = []

    def record(
        self,
        step: int,
        result: EvaluationResult,
        label: Optional[str] = None,
    ) -> None:
        """Record an EvaluationResult at a given training step.

        Raises ValueError if a result for this step already exists.
        """
        normalized_step = int(step)
        if any(entry.step == normalized_step for entry in self._entries):
            raise ValueError(f"step already recorded: {normalized_step}")
        self._entries.append(
            CheckpointEntry(step=normalized_step, result=result, label=label)
        )
        self._entries.sort(key=lambda entry: entry.step)

    def steps(self) -> list[int]:
        """Return recorded steps in ascending order."""
        return [entry.step for entry in self._entries]

    def get(self, step: int) -> EvaluationResult:
        """Return the result recorded at a given step.

        Raises KeyError if step not recorded.
        """
        normalized_step = int(step)
        for entry in self._entries:
            if entry.step == normalized_step:
                return entry.result
        raise KeyError(f"step not recorded: {normalized_step}")

    def metric_values(self, metric_name: str) -> dict[int, float]:
        """Return {step: value} for a metric across all recorded steps.

        Steps where the metric was not computed or had an error return nan.
        """
        values: dict[int, float] = {}
        for entry in self._entries:
            values[entry.step] = _metric_value(entry.result, metric_name)
        return values

    def trend(self, metric_name: str) -> float:
        """Return the linear trend (slope) of a metric over steps.

        Uses least-squares linear regression of metric_values against steps.
        Positive slope means the metric is increasing over training.
        Returns nan if fewer than 2 finite values exist.
        For metrics where lower is better, a negative trend is good.
        For metrics where higher is better, a positive trend is good.
        The caller is responsible for interpreting direction.
        """
        values_by_step = self.metric_values(metric_name)
        finite = [
            (step, value)
            for step, value in values_by_step.items()
            if np.isfinite(value)
        ]
        if len(finite) < 2:
            return float("nan")
        steps = np.asarray([step for step, _ in finite], dtype=np.float64)
        values = np.asarray([value for _, value in finite], dtype=np.float64)
        return float(np.polyfit(steps, values, deg=1)[0])

    def best_step(
        self,
        metric_name: str,
        *,
        higher_is_better: bool = False,
    ) -> int:
        """Return the step with the best value for a given metric.

        Raises ValueError if no finite values exist for this metric.
        """
        finite = [
            (step, value)
            for step, value in self.metric_values(metric_name).items()
            if np.isfinite(value)
        ]
        if not finite:
            raise ValueError(f"no finite values recorded for metric: {metric_name}")
        key = (lambda item: item[1]) if higher_is_better else (lambda item: -item[1])
        return max(finite, key=key)[0]

    def to_markdown(
        self,
        metrics: Optional[list[str]] = None,
    ) -> str:
        """Return a Markdown table with steps as rows and metrics as columns.

        If metrics is None, include all metrics seen across all steps.
        Missing values are rendered as '-'.
        """
        metric_names = list(metrics) if metrics is not None else self._metric_names()
        header = ["Step", "Label", *metric_names]
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * len(header)) + " |",
        ]
        for entry in self._entries:
            row = [str(entry.step), entry.label or "-"]
            values = self._metric_values_by_entry(entry)
            for metric_name in metric_names:
                value = values.get(metric_name, float("nan"))
                row.append(_format_float(value) if np.isfinite(value) else "-")
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible representation."""
        return {
            "checkpoints": [
                {
                    "step": entry.step,
                    "label": entry.label,
                    "result": entry.result.to_dict(),
                }
                for entry in self._entries
            ]
        }

    def to_json(self) -> str:
        """Return JSON string."""
        return json.dumps(self.to_dict(), allow_nan=False, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> EvaluationHistory:
        """Reconstruct from JSON produced by to_json()."""
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("EvaluationHistory JSON must decode to an object")
        checkpoints = data.get("checkpoints")
        if not isinstance(checkpoints, list):
            raise ValueError("EvaluationHistory JSON must contain a checkpoints list")
        history = cls()
        for item in checkpoints:
            if not isinstance(item, dict):
                raise ValueError("EvaluationHistory checkpoints must contain objects")
            raw_result = item.get("result")
            if not isinstance(raw_result, dict):
                raise ValueError("EvaluationHistory checkpoint result must be an object")
            label = item.get("label")
            history.record(
                step=int(item["step"]),
                result=EvaluationResult.from_dict(raw_result),
                label=str(label) if label is not None else None,
            )
        return history

    def _metric_values_by_entry(self, entry: CheckpointEntry) -> dict[str, float]:
        """Return metric values for one checkpoint keyed by metric name."""
        return {
            metric.name: (
                float("nan")
                if "error" in metric.metadata
                else float(metric.value)
            )
            for metric in entry.result.results
        }

    def _metric_names(self) -> list[str]:
        names: list[str] = []
        for entry in self._entries:
            for metric in entry.result.results:
                if metric.name not in names:
                    names.append(metric.name)
        return names


def _metric_value(result: EvaluationResult, metric_name: str) -> float:
    for metric in result.results:
        if metric.name == metric_name:
            if "error" in metric.metadata:
                return float("nan")
            return float(metric.value)
    return float("nan")


def _format_float(value: float) -> str:
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    return f"{value:.6g}"

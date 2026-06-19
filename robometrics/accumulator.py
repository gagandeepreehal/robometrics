"""Streaming metric accumulation helpers."""

from __future__ import annotations

from threading import RLock
from typing import Optional

from robometrics._version import __version__
from robometrics.evaluator import _aggregate_metric_values
from robometrics.results import EvaluationResult


class MetricAccumulator:
    """Stateful per-metric accumulator for streaming evaluation.

    Follows the update()/compute() pattern used by TorchMetrics.
    Designed for use inside training loops where you want to accumulate
    metric values across batches and compute aggregate statistics at
    epoch end without storing all individual results in memory.

    Usage:
        acc = MetricAccumulator(metrics=["ade", "fde"])
        for batch in dataloader:
            result = evaluator.evaluate(...)
            acc.update(result)
        summary = acc.compute()
        acc.reset()
    """

    def __init__(
        self,
        metrics: Optional[list[str]] = None,
    ) -> None:
        """
        Args:
            metrics: optional list of metric names to track. If None,
                     all metrics seen during update() calls are tracked.
        """
        self._lock = RLock()
        self._tracked = None if metrics is None else set(metrics)
        self._tracked_order = list(metrics or [])
        self._values: dict[str, list[float]] = {}
        self._units: dict[str, str] = {}
        self._metadata_templates: dict[str, dict[str, object]] = {}
        self._sample_count = 0

    def update(self, result: EvaluationResult) -> None:
        """Ingest one EvaluationResult.

        Only metrics in self._tracked are stored if metrics was specified
        at construction. Otherwise all metrics are tracked.
        Non-finite metric values are stored as-is and excluded from
        aggregate statistics at compute() time, consistent with
        evaluate_dataset() behavior.
        """
        with self._lock:
            self._sample_count += 1
            for metric in result.results:
                if self._tracked is not None and metric.name not in self._tracked:
                    continue
                if metric.name not in self._values:
                    self._values[metric.name] = []
                    if self._tracked is None:
                        self._tracked_order.append(metric.name)
                self._values[metric.name].append(float(metric.value))
                self._units.setdefault(metric.name, metric.unit)
                self._metadata_templates.setdefault(
                    metric.name,
                    {
                        "category": metric.metadata.get("category"),
                        "description": metric.metadata.get("description"),
                        "reference": metric.metadata.get("reference"),
                        "is_novel": metric.metadata.get("is_novel"),
                    },
                )

    def compute(self) -> EvaluationResult:
        """Return an aggregated EvaluationResult over all updated results.

        Produces the same aggregate structure as Evaluator.evaluate_dataset():
        one MetricResult per metric with mean, std, min, max, finite_count,
        and sample_count in metadata. Returns an EvaluationResult with
        metadata["accumulator"] = True and metadata["sample_count"] = N.
        Raises RuntimeError if called before any update().
        """
        with self._lock:
            if self._sample_count == 0:
                raise RuntimeError("MetricAccumulator.compute() called before update()")

            results = []
            for name in self._tracked_order:
                values = self._values.get(name)
                if not values:
                    continue
                metadata_template = dict(self._metadata_templates.get(name, {}))
                metadata_template["sample_count"] = self._sample_count
                results.append(
                    _aggregate_metric_values(
                        name=name,
                        values=list(values),
                        unit=self._units.get(name, ""),
                        metadata_template=metadata_template,
                    )
                )

            return EvaluationResult(
                results=results,
                metadata={
                    "robometrics_version": __version__,
                    "accumulator": True,
                    "sample_count": self._sample_count,
                },
            )

    def reset(self) -> None:
        """Clear all accumulated values. Resets sample count to 0."""
        with self._lock:
            self._values.clear()
            self._units.clear()
            self._metadata_templates.clear()
            if self._tracked is None:
                self._tracked_order.clear()
            self._sample_count = 0

    @property
    def sample_count(self) -> int:
        """Number of EvaluationResult objects ingested since last reset."""
        with self._lock:
            return self._sample_count

    @property
    def tracked_metrics(self) -> list[str]:
        """Metric names currently being tracked."""
        with self._lock:
            return list(self._tracked_order)

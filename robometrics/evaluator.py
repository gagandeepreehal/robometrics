"""Lightweight local evaluator for named metric functions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional, Union

import numpy as np

from robometrics._version import __version__
from robometrics.registry import MetricDefinition, MetricRegistry, registry
from robometrics.results import EvaluationResult, MetricResult


class EvaluationInputError(ValueError):
    """Raised when an evaluation request cannot be constructed."""


class Evaluator:
    """Evaluate registered RoboMetrics metrics against local Python inputs."""

    def __init__(self, metric_registry: Optional[MetricRegistry] = None) -> None:
        self.registry = metric_registry or registry

    def evaluate(
        self,
        *,
        prediction: Optional[Any] = None,
        ground_truth: Optional[Any] = None,
        metrics: Optional[Union[str, Sequence[str]]] = "all",
        categories: Optional[Union[str, Sequence[str]]] = None,
        thresholds: Optional[Mapping[str, float]] = None,
        metric_kwargs: Optional[Mapping[str, Mapping[str, Any]]] = None,
        **inputs: Any,
    ) -> EvaluationResult:
        """Run selected metrics and return an EvaluationResult.

        ``prediction`` and ``ground_truth`` are mapped onto the common argument
        names used by built-in metric functions. Additional metric-specific
        inputs, such as ``dt`` or ``actor_trajs``, can be supplied as keyword
        arguments. Metrics that return arrays are reduced to scalar results:
        vector arrays use mean row-wise norm, and other arrays use the mean.
        The raw array and reduction name are stored in each result's metadata.
        """
        _validate_common_array(prediction, name="prediction", allowed_ndims=(2, 3))
        _validate_common_array(ground_truth, name="ground_truth", allowed_ndims=(2,))
        input_values = _build_inputs(
            prediction=prediction,
            ground_truth=ground_truth,
            inputs=inputs,
        )
        if not input_values:
            raise EvaluationInputError("at least one evaluation input is required")

        selected, automatic_selection = self._select_metrics(metrics=metrics, categories=categories)
        threshold_map = self._normalize_thresholds(thresholds)
        metric_kwargs_map = self._normalize_metric_kwargs(metric_kwargs)

        results: list[MetricResult] = []
        skipped_metrics: list[str] = []
        for metric in selected:
            if not metric.is_compatible(input_values):
                if automatic_selection:
                    skipped_metrics.append(metric.name)
                    continue
                results.append(
                    _error_result(
                        metric,
                        "provided inputs are not compatible with this metric",
                    )
                )
                continue

            try:
                result = self._run_metric(metric, input_values, metric_kwargs_map)
            except Exception as exc:  # noqa: BLE001 - failures must be stored per metric.
                result = _error_result(metric, str(exc), exc)

            threshold = threshold_map.get(metric.name)
            if threshold is not None and "error" not in result.metadata:
                result.threshold = threshold
                result.passed = bool(result.value <= threshold)
            results.append(result)

        if not results and selected:
            raise EvaluationInputError(_no_compatible_metrics_message(selected, input_values))

        return EvaluationResult(
            results=results,
            metadata={
                "robometrics_version": __version__,
                "selected_metrics": [metric.name for metric in selected],
                "skipped_metrics": skipped_metrics,
                "categories": _category_list(categories),
            },
        )

    def _select_metrics(
        self,
        *,
        metrics: Optional[Union[str, Sequence[str]]],
        categories: Optional[Union[str, Sequence[str]]],
    ) -> tuple[list[MetricDefinition], bool]:
        requested_categories = _category_list(categories)
        if requested_categories:
            known_categories = {category.lower() for category in self.registry.categories()}
            unknown_categories = [
                category
                for category in requested_categories
                if category.lower() not in known_categories
            ]
            if unknown_categories:
                raise EvaluationInputError(
                    "unknown metric categories: " + ", ".join(unknown_categories)
                )

        automatic_selection = metrics is None or (
            isinstance(metrics, str) and _normalize_name(metrics) == "all"
        )

        if automatic_selection:
            selected = self.registry.list_metrics()
            if requested_categories:
                requested = {category.lower() for category in requested_categories}
                selected = [metric for metric in selected if metric.category.lower() in requested]
            return selected, True

        requested_names = [metrics] if isinstance(metrics, str) else list(metrics or ())

        selected = []
        seen: set[str] = set()
        for name in requested_names:
            metric = self.registry.get(name)
            if metric.name not in seen:
                selected.append(metric)
                seen.add(metric.name)
        return selected, False

    def _normalize_thresholds(
        self,
        thresholds: Optional[Mapping[str, float]],
    ) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for name, threshold in (thresholds or {}).items():
            metric = self.registry.get(name)
            value = float(threshold)
            if not np.isfinite(value):
                raise EvaluationInputError(f"threshold for {name} must be finite")
            normalized[metric.name] = value
        return normalized

    def _normalize_metric_kwargs(
        self,
        metric_kwargs: Optional[Mapping[str, Mapping[str, Any]]],
    ) -> dict[str, dict[str, Any]]:
        normalized: dict[str, dict[str, Any]] = {}
        for name, kwargs in (metric_kwargs or {}).items():
            metric = self.registry.get(name)
            normalized[metric.name] = dict(kwargs)
        return normalized

    def _run_metric(
        self,
        metric: MetricDefinition,
        inputs: Mapping[str, Any],
        metric_kwargs: Mapping[str, Mapping[str, Any]],
    ) -> MetricResult:
        kwargs: dict[str, Any] = {key: inputs[key] for key in metric.required_inputs}
        kwargs.update(metric.default_kwargs)
        for key in metric.default_kwargs:
            if key in inputs:
                kwargs[key] = inputs[key]
        kwargs.update(metric_kwargs.get(metric.name, {}))

        raw_value = metric.fn(**kwargs)
        return _result_from_raw(metric, raw_value)


def _build_inputs(
    *,
    prediction: Optional[Any],
    ground_truth: Optional[Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    values = {key: value for key, value in inputs.items() if value is not None}
    if prediction is not None:
        values["prediction"] = prediction
        for alias in ("pred", "predictions", "traj", "trajectory", "ego_traj"):
            values.setdefault(alias, prediction)
    if ground_truth is not None:
        values["ground_truth"] = ground_truth
        for alias in ("gt", "ref", "reference"):
            values.setdefault(alias, ground_truth)
    return values


def _validate_common_array(
    value: Optional[Any],
    *,
    name: str,
    allowed_ndims: tuple[int, ...],
) -> None:
    if value is None:
        return
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise EvaluationInputError(f"{name} must be a numeric array-like value") from exc

    if arr.ndim not in allowed_ndims:
        allowed = " or ".join(str(ndim) for ndim in allowed_ndims)
        raise EvaluationInputError(f"{name} must have {allowed} dimensions")
    if arr.ndim == 2 and (arr.shape[0] == 0 or arr.shape[1] not in (2, 3)):
        raise EvaluationInputError(f"{name} must be a non-empty Nx2 or Nx3 array")
    if arr.ndim == 3 and (arr.shape[0] == 0 or arr.shape[1] == 0 or arr.shape[2] not in (2, 3)):
        raise EvaluationInputError(f"{name} must be a non-empty KxTx2 or KxTx3 array")
    if not np.all(np.isfinite(arr)):
        raise EvaluationInputError(f"{name} must contain only finite values")


def _result_from_raw(metric: MetricDefinition, raw_value: Any) -> MetricResult:
    if isinstance(raw_value, MetricResult):
        metadata = dict(raw_value.metadata)
        metadata.setdefault("category", metric.category)
        metadata.setdefault("description", metric.description)
        return MetricResult(
            name=metric.name,
            value=raw_value.value,
            unit=raw_value.unit or metric.unit,
            passed=raw_value.passed,
            threshold=raw_value.threshold,
            metadata=metadata,
        )

    value, metadata = _coerce_metric_value(raw_value)
    metadata.setdefault("category", metric.category)
    metadata.setdefault("description", metric.description)
    return MetricResult(name=metric.name, value=value, unit=metric.unit, metadata=metadata)


def _coerce_metric_value(raw_value: Any) -> tuple[float, dict[str, Any]]:
    arr = np.asarray(raw_value, dtype=np.float64)
    if arr.ndim == 0:
        return float(arr), {}
    if arr.size == 0:
        return float("nan"), {"reduction": "empty", "raw_value": []}

    if arr.ndim >= 2 and arr.shape[-1] in (2, 3):
        reduced = np.linalg.norm(arr, axis=-1)
        reduction = "mean_norm"
    else:
        reduced = arr
        reduction = "mean"

    finite_values = reduced[np.isfinite(reduced)]
    value = float(np.mean(finite_values)) if finite_values.size else float("nan")
    return (
        value,
        {
            "reduction": reduction,
            "shape": list(arr.shape),
            "value_count": int(arr.size),
            "raw_value": arr.tolist(),
        },
    )


def _error_result(
    metric: MetricDefinition,
    message: str,
    exc: Optional[Exception] = None,
) -> MetricResult:
    metadata: dict[str, Any] = {
        "category": metric.category,
        "description": metric.description,
        "error": message,
    }
    if exc is not None:
        metadata["error_type"] = type(exc).__name__
    return MetricResult(
        name=metric.name,
        value=float("nan"),
        unit=metric.unit,
        passed=False,
        metadata=metadata,
    )


def _no_compatible_metrics_message(
    selected: Sequence[MetricDefinition],
    inputs: Mapping[str, Any],
) -> str:
    missing_by_metric = {
        metric.name: [
            key
            for key in metric.required_inputs
            if key not in inputs or inputs[key] is None
        ]
        for metric in selected
    }
    missing_inputs = sorted({key for keys in missing_by_metric.values() for key in keys})
    if missing_inputs:
        details = "; ".join(
            f"{name} missing {', '.join(keys)}"
            for name, keys in missing_by_metric.items()
            if keys
        )
        return "missing required inputs: " + ", ".join(missing_inputs) + f" ({details})"
    return "no compatible metrics found for the provided inputs"


def _category_list(categories: Optional[Union[str, Sequence[str]]]) -> list[str]:
    if categories is None:
        return []
    if isinstance(categories, str):
        return [categories]
    return list(categories)


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")

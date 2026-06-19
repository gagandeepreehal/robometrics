"""Lightweight local evaluator for named metric functions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional, Union

import numpy as np

from robometrics._version import __version__
from robometrics.registry import MetricDefinition, MetricRegistry, registry
from robometrics.results import EvaluationResult, MetricResult
from robometrics.schemas import Trajectory


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

    def evaluate_dataset(
        self,
        *,
        predictions: Sequence[Any],
        ground_truths: Sequence[Any],
        metrics: Optional[Union[str, Sequence[str]]] = "all",
        categories: Optional[Union[str, Sequence[str]]] = None,
        thresholds: Optional[Mapping[str, float]] = None,
        metric_kwargs: Optional[Mapping[str, Mapping[str, Any]]] = None,
        bootstrap_ci: Optional[int] = None,
        ci_alpha: float = 0.05,
        bootstrap_seed: Optional[int] = 0,
        **inputs: Any,
    ) -> EvaluationResult:
        """Evaluate matching prediction/ground-truth sequences and aggregate by metric."""
        _validate_confidence_interval_config(
            bootstrap_ci=bootstrap_ci,
            ci_alpha=ci_alpha,
            bootstrap_seed=bootstrap_seed,
        )
        if len(predictions) != len(ground_truths):
            raise EvaluationInputError("predictions and ground_truths must have the same length")
        if len(predictions) == 0:
            raise EvaluationInputError("dataset evaluation requires at least one sample")

        sample_results = [
            self.evaluate(
                prediction=prediction,
                ground_truth=ground_truth,
                metrics=metrics,
                categories=categories,
                thresholds=thresholds,
                metric_kwargs=metric_kwargs,
                **inputs,
            )
            for prediction, ground_truth in zip(predictions, ground_truths)
        ]
        threshold_map = self._normalize_thresholds(thresholds)
        return _aggregate_dataset_results(
            sample_results,
            thresholds=threshold_map,
            categories=_category_list(categories),
            bootstrap_ci=bootstrap_ci,
            ci_alpha=ci_alpha,
            bootstrap_seed=bootstrap_seed,
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
    values = {
        key: _coerce_input_value(value)
        for key, value in inputs.items()
        if value is not None
    }
    if prediction is not None:
        prediction_value = _coerce_input_value(prediction)
        values["prediction"] = prediction_value
        for alias in ("pred", "predictions", "traj", "trajectory", "ego_traj"):
            values.setdefault(alias, prediction_value)
    if ground_truth is not None:
        ground_truth_value = _coerce_input_value(ground_truth)
        values["ground_truth"] = ground_truth_value
        for alias in ("gt", "ref", "reference"):
            values.setdefault(alias, ground_truth_value)
    return values


def _coerce_input_value(value: Any) -> Any:
    if isinstance(value, Trajectory):
        return value.array()
    if isinstance(value, list) and any(isinstance(item, Trajectory) for item in value):
        return [
            item.array() if isinstance(item, Trajectory) else item
            for item in value
        ]
    if isinstance(value, tuple) and any(isinstance(item, Trajectory) for item in value):
        return tuple(item.array() if isinstance(item, Trajectory) else item for item in value)
    return value


def _validate_common_array(
    value: Optional[Any],
    *,
    name: str,
    allowed_ndims: tuple[int, ...],
) -> None:
    if value is None:
        return
    try:
        arr = np.asarray(_coerce_input_value(value), dtype=np.float64)
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
        metadata.update(_metric_metadata(metric, metadata))
        return MetricResult(
            name=metric.name,
            value=raw_value.value,
            unit=raw_value.unit or metric.unit,
            passed=raw_value.passed,
            threshold=raw_value.threshold,
            metadata=metadata,
        )

    value, metadata = _coerce_metric_value(raw_value)
    metadata.update(_metric_metadata(metric, metadata))
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
        "error": message,
    }
    metadata.update(_metric_metadata(metric, metadata))
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


def _aggregate_dataset_results(
    sample_results: Sequence[EvaluationResult],
    *,
    thresholds: Mapping[str, float],
    categories: list[str],
    bootstrap_ci: Optional[int] = None,
    ci_alpha: float = 0.05,
    bootstrap_seed: Optional[int] = 0,
) -> EvaluationResult:
    metric_names: list[str] = []
    for sample in sample_results:
        for result in sample.results:
            if result.name not in metric_names:
                metric_names.append(result.name)

    aggregate_results = []
    for name in metric_names:
        per_sample = [
            sample.results[index]
            for sample in sample_results
            for index, result in enumerate(sample.results)
            if result.name == name
        ]
        first = per_sample[0]
        metric = _aggregate_metric_values(
            name=name,
            values=[result.value for result in per_sample],
            unit=first.unit,
            metadata_template={
                "category": first.metadata.get("category"),
                "description": first.metadata.get("description"),
                "reference": first.metadata.get("reference"),
                "is_novel": first.metadata.get("is_novel"),
                "sample_count": len(sample_results),
                "errors": [
                    result.metadata.get("error")
                    for result in per_sample
                    if "error" in result.metadata
                ],
            },
        )
        threshold = thresholds.get(name)
        metric.threshold = threshold
        metric.passed = None if threshold is None else bool(metric.value <= threshold)
        aggregate_results.append(metric)

    if bootstrap_ci is not None:
        rng = np.random.default_rng(bootstrap_seed)
        for metric in aggregate_results:
            _add_bootstrap_confidence_interval(
                metric,
                bootstrap_ci=bootstrap_ci,
                ci_alpha=ci_alpha,
                rng=rng,
            )
            metric.metadata["bootstrap_seed"] = bootstrap_seed

    return EvaluationResult(
        results=aggregate_results,
        metadata={
            "robometrics_version": __version__,
            "sample_count": len(sample_results),
            "categories": categories,
            "dataset": True,
            "bootstrap_seed": bootstrap_seed if bootstrap_ci is not None else None,
        },
    )


def _validate_confidence_interval_config(
    *,
    bootstrap_ci: Optional[int],
    ci_alpha: float,
    bootstrap_seed: Optional[int],
) -> None:
    if not 0.0 < float(ci_alpha) < 0.5:
        raise EvaluationInputError("ci_alpha must be in the open interval (0, 0.5)")
    if bootstrap_ci is None:
        return
    if not isinstance(bootstrap_ci, int) or isinstance(bootstrap_ci, bool) or bootstrap_ci < 100:
        raise EvaluationInputError(
            "bootstrap_ci must be at least 100 for reliable confidence intervals"
        )
    if (
        bootstrap_seed is not None
        and (not isinstance(bootstrap_seed, int) or isinstance(bootstrap_seed, bool))
    ):
        raise EvaluationInputError("bootstrap_seed must be an integer or None")


def _add_bootstrap_confidence_interval(
    metric: MetricResult,
    *,
    bootstrap_ci: int,
    ci_alpha: float,
    rng: np.random.Generator,
) -> None:
    values = np.asarray(metric.metadata.get("values", []), dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    if finite_values.size < 2:
        return

    indices = rng.integers(
        low=0,
        high=finite_values.size,
        size=(bootstrap_ci, finite_values.size),
    )
    means = np.mean(finite_values[indices], axis=1)
    percentiles = np.asarray(
        np.percentile(
            means,
            [ci_alpha / 2.0 * 100.0, (1.0 - ci_alpha / 2.0) * 100.0],
        ),
        dtype=np.float64,
    )
    lower = float(percentiles[0])
    upper = float(percentiles[1])
    metric.metadata["ci_lower"] = float(lower)
    metric.metadata["ci_upper"] = float(upper)
    metric.metadata["ci_alpha"] = float(ci_alpha)
    metric.metadata["bootstrap_n"] = int(bootstrap_ci)


def _aggregate_metric_values(
    name: str,
    values: Sequence[float],
    unit: str,
    metadata_template: Mapping[str, Any],
) -> MetricResult:
    """Aggregate scalar metric values into one dataset-style MetricResult."""
    raw_values = [float(value) for value in values]
    finite_values = np.asarray(
        [value for value in raw_values if np.isfinite(value)],
        dtype=np.float64,
    )
    value = float(np.mean(finite_values)) if finite_values.size else float("nan")
    metadata = dict(metadata_template)
    metadata.update(
        {
            "sample_count": int(metadata.get("sample_count", len(raw_values))),
            "finite_count": int(finite_values.size),
            "mean": value if finite_values.size else None,
            "std": float(np.std(finite_values)) if finite_values.size else None,
            "min": float(np.min(finite_values)) if finite_values.size else None,
            "max": float(np.max(finite_values)) if finite_values.size else None,
            "values": raw_values,
            "errors": list(metadata.get("errors", [])),
        }
    )
    return MetricResult(name=name, value=value, unit=unit, metadata=metadata)


def _metric_metadata(metric: MetricDefinition, existing: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(existing)
    metadata.setdefault("category", metric.category)
    metadata.setdefault("description", metric.description)
    metadata.setdefault("reference", metric.reference)
    metadata.setdefault("is_novel", metric.is_novel)
    metadata.setdefault("higher_is_better", metric.higher_is_better)
    return metadata

"""Dataset validation helpers for lightweight trajectory files."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

PathLike = Union[str, Path]

_TIMESTAMP_FIELDS = ("timestamp", "time", "t")


@dataclass(frozen=True)
class ValidationIssue:
    """One dataset validation issue."""

    code: str
    message: str
    field: Optional[str] = None
    index: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible issue payload."""
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.field is not None:
            payload["field"] = self.field
        if self.index is not None:
            payload["index"] = self.index
        return payload


@dataclass(frozen=True)
class DatasetValidationResult:
    """Validation report for one CSV or JSON trajectory-style dataset."""

    path: str
    format: str
    row_count: int = 0
    dimensions: Optional[int] = None
    has_timestamps: bool = False
    issues: list[ValidationIssue] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Return True when no validation issues were found."""
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible validation report."""
        return {
            "path": self.path,
            "format": self.format,
            "passed": self.passed,
            "row_count": self.row_count,
            "dimensions": self.dimensions,
            "has_timestamps": self.has_timestamps,
            "issues": [issue.to_dict() for issue in self.issues],
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Return a strict JSON validation report."""
        return json.dumps(self.to_dict(), allow_nan=False, sort_keys=True)

    def to_text(self) -> str:
        """Return a concise human-readable validation summary."""
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"{status}: {self.path}",
            f"format: {self.format}",
            f"rows: {self.row_count}",
            f"dimensions: {self.dimensions if self.dimensions is not None else '-'}",
            f"timestamps: {self.has_timestamps}",
        ]
        if not self.issues:
            lines.append("issues: none")
            return "\n".join(lines)

        lines.append("issues:")
        for issue in self.issues:
            location = []
            if issue.index is not None:
                location.append(f"row {issue.index}")
            if issue.field is not None:
                location.append(issue.field)
            prefix = f"  - {issue.code}"
            if location:
                prefix += f" ({', '.join(location)})"
            lines.append(f"{prefix}: {issue.message}")
        return "\n".join(lines)


def validate_dataset(path: PathLike) -> DatasetValidationResult:
    """Validate a trajectory-style CSV or JSON file."""
    dataset_path = Path(path)
    suffix = dataset_path.suffix.lower()
    if not dataset_path.exists():
        return DatasetValidationResult(
            path=str(dataset_path),
            format=suffix.lstrip(".") or "unknown",
            issues=[
                ValidationIssue(
                    code="missing_file",
                    message=f"dataset file does not exist: {dataset_path}",
                )
            ],
        )
    if not dataset_path.is_file():
        return DatasetValidationResult(
            path=str(dataset_path),
            format=suffix.lstrip(".") or "unknown",
            issues=[
                ValidationIssue(
                    code="not_a_file",
                    message=f"dataset path is not a file: {dataset_path}",
                )
            ],
        )
    if suffix == ".csv":
        return _validate_csv(dataset_path)
    if suffix == ".json":
        return _validate_json(dataset_path)
    return DatasetValidationResult(
        path=str(dataset_path),
        format=suffix.lstrip(".") or "unknown",
        issues=[
            ValidationIssue(
                code="unsupported_format",
                message="supported formats are .csv and .json",
            )
        ],
    )


def _validate_csv(path: Path) -> DatasetValidationResult:
    issues: list[ValidationIssue] = []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
    except csv.Error as exc:
        return DatasetValidationResult(
            path=str(path),
            format="csv",
            issues=[ValidationIssue(code="parse_error", message=str(exc))],
        )
    except OSError as exc:
        return DatasetValidationResult(
            path=str(path),
            format="csv",
            issues=[ValidationIssue(code="read_error", message=str(exc))],
        )

    missing_fields = [name for name in ("x", "y") if name not in fieldnames]
    for missing_field in missing_fields:
        issues.append(
            ValidationIssue(
                code="missing_required_field",
                message=f"CSV trajectory data requires an {missing_field!r} column",
                field=missing_field,
            )
        )

    if not rows:
        issues.append(ValidationIssue(code="empty_trajectory", message="trajectory has no rows"))

    timestamp_field = _first_present(fieldnames, _TIMESTAMP_FIELDS)
    z_present = "z" in fieldnames
    dimensions_seen: set[int] = set()
    timestamps: list[float] = []
    if not missing_fields:
        for row_index, row in enumerate(rows):
            dimensions = _validate_coordinate_row(
                row,
                index=row_index,
                issues=issues,
                z_present=z_present,
            )
            if dimensions is not None:
                dimensions_seen.add(dimensions)
            if timestamp_field is not None:
                value = _finite_float(
                    row.get(timestamp_field, ""),
                    field=timestamp_field,
                    index=row_index,
                    issues=issues,
                )
                if value is not None:
                    timestamps.append(value)

    issues.extend(_dimension_issues(dimensions_seen))
    if timestamp_field is not None:
        issues.extend(_timestamp_issues(timestamps))

    dimensions = next(iter(dimensions_seen)) if len(dimensions_seen) == 1 else None
    return DatasetValidationResult(
        path=str(path),
        format="csv",
        row_count=len(rows),
        dimensions=dimensions,
        has_timestamps=timestamp_field is not None,
        issues=issues,
        metadata={"columns": fieldnames},
    )


def _validate_json(path: Path) -> DatasetValidationResult:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return DatasetValidationResult(
            path=str(path),
            format="json",
            issues=[ValidationIssue(code="parse_error", message=str(exc))],
        )
    except OSError as exc:
        return DatasetValidationResult(
            path=str(path),
            format="json",
            issues=[ValidationIssue(code="read_error", message=str(exc))],
        )

    records = _json_records(payload)
    if not isinstance(records, list):
        return DatasetValidationResult(
            path=str(path),
            format="json",
            issues=[
                ValidationIssue(
                    code="missing_required_field",
                    message=(
                        "JSON trajectory data must be a list or contain "
                        "'trajectory' or 'points'"
                    ),
                )
            ],
        )

    issues: list[ValidationIssue] = []
    if not records:
        issues.append(ValidationIssue(code="empty_trajectory", message="trajectory has no rows"))

    dimensions_seen: set[int] = set()
    timestamps: list[float] = []
    has_timestamps = False
    for index, record in enumerate(records):
        dimensions = _validate_json_record(record, index=index, issues=issues)
        if dimensions is not None:
            dimensions_seen.add(dimensions)
        timestamp = _json_timestamp(record)
        if timestamp is not None:
            has_timestamps = True
            value = _finite_float(timestamp, field="timestamp", index=index, issues=issues)
            if value is not None:
                timestamps.append(value)

    issues.extend(_dimension_issues(dimensions_seen))
    if has_timestamps:
        issues.extend(_timestamp_issues(timestamps))

    dimensions = next(iter(dimensions_seen)) if len(dimensions_seen) == 1 else None
    return DatasetValidationResult(
        path=str(path),
        format="json",
        row_count=len(records),
        dimensions=dimensions,
        has_timestamps=has_timestamps,
        issues=issues,
        metadata={"container": _json_container_name(payload)},
    )


def _validate_coordinate_row(
    row: dict[str, str],
    *,
    index: int,
    issues: list[ValidationIssue],
    z_present: bool,
) -> Optional[int]:
    dimensions = 2
    for coord_field in ("x", "y"):
        _finite_float(
            row.get(coord_field, ""),
            field=coord_field,
            index=index,
            issues=issues,
        )
    if z_present:
        z_value = row.get("z", "")
        if z_value == "":
            issues.append(
                ValidationIssue(
                    code="inconsistent_dimensions",
                    message="z is present as a column but missing for this row",
                    field="z",
                    index=index,
                )
            )
            return None
        _finite_float(z_value, field="z", index=index, issues=issues)
        dimensions = 3
    return dimensions


def _validate_json_record(
    record: Any,
    *,
    index: int,
    issues: list[ValidationIssue],
) -> Optional[int]:
    if isinstance(record, dict):
        missing = [field for field in ("x", "y") if field not in record]
        for field in missing:
            issues.append(
                ValidationIssue(
                    code="missing_required_field",
                    message=f"JSON trajectory point requires {field!r}",
                    field=field,
                    index=index,
                )
            )
        if missing:
            return None
        for field in ("x", "y"):
            _finite_float(record[field], field=field, index=index, issues=issues)
        if "z" in record:
            _finite_float(record["z"], field="z", index=index, issues=issues)
            return 3
        return 2

    if isinstance(record, list):
        if len(record) not in (2, 3):
            issues.append(
                ValidationIssue(
                    code="inconsistent_dimensions",
                    message="coordinate list must have length 2 or 3",
                    index=index,
                )
            )
            return None
        for coord_index, value in enumerate(record):
            _finite_float(
                value,
                field=f"coord[{coord_index}]",
                index=index,
                issues=issues,
            )
        return len(record)

    issues.append(
        ValidationIssue(
            code="invalid_record",
            message="trajectory record must be an object or coordinate list",
            index=index,
        )
    )
    return None


def _finite_float(
    raw: Any,
    *,
    field: str,
    index: int,
    issues: list[ValidationIssue],
) -> Optional[float]:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        issues.append(
            ValidationIssue(
                code="invalid_numeric_value",
                message=f"{field} must be numeric",
                field=field,
                index=index,
            )
        )
        return None
    if not math.isfinite(value):
        issues.append(
            ValidationIssue(
                code="invalid_numeric_value",
                message=f"{field} must be finite",
                field=field,
                index=index,
            )
        )
        return None
    return value


def _timestamp_issues(timestamps: list[float]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for index in range(1, len(timestamps)):
        if timestamps[index] <= timestamps[index - 1]:
            issues.append(
                ValidationIssue(
                    code="non_monotonic_timestamps",
                    message="timestamps must be strictly increasing",
                    field="timestamp",
                    index=index,
                )
            )
            break
    return issues


def _dimension_issues(dimensions_seen: set[int]) -> list[ValidationIssue]:
    if len(dimensions_seen) <= 1:
        return []
    return [
        ValidationIssue(
            code="inconsistent_dimensions",
            message="trajectory mixes 2D and 3D points",
        )
    ]


def _json_records(payload: Any) -> Any:
    if isinstance(payload, dict):
        if "trajectory" in payload:
            return payload["trajectory"]
        if "points" in payload:
            return payload["points"]
    return payload


def _json_container_name(payload: Any) -> str:
    if isinstance(payload, dict):
        if "trajectory" in payload:
            return "trajectory"
        if "points" in payload:
            return "points"
        return "object"
    if isinstance(payload, list):
        return "list"
    return type(payload).__name__


def _json_timestamp(record: Any) -> Any:
    if not isinstance(record, dict):
        return None
    for timestamp_field in _TIMESTAMP_FIELDS:
        if timestamp_field in record:
            return record[timestamp_field]
    return None


def _first_present(values: list[str], candidates: tuple[str, ...]) -> Optional[str]:
    for candidate in candidates:
        if candidate in values:
            return candidate
    return None

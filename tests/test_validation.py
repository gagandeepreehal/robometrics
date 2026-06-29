from __future__ import annotations

import json

from robometrics.validation import DatasetValidationResult, ValidationIssue, validate_dataset


def test_validation_result_serializes_issue_locations() -> None:
    result = DatasetValidationResult(
        path="bad.csv",
        format="csv",
        row_count=1,
        issues=[
            ValidationIssue(
                code="invalid_numeric_value",
                message="x must be numeric",
                field="x",
                index=0,
            )
        ],
    )

    payload = result.to_dict()
    assert payload["passed"] is False
    assert payload["issues"] == [
        {
            "code": "invalid_numeric_value",
            "message": "x must be numeric",
            "field": "x",
            "index": 0,
        }
    ]
    assert json.loads(result.to_json())["issues"][0]["field"] == "x"
    assert "FAIL: bad.csv" in result.to_text()
    assert "invalid_numeric_value (row 0, x)" in result.to_text()


def test_validate_dataset_rejects_missing_directory_and_unsupported_paths(tmp_path) -> None:
    missing = validate_dataset(tmp_path / "missing.json")
    assert missing.issues[0].code == "missing_file"

    directory = validate_dataset(tmp_path)
    assert directory.issues[0].code == "not_a_file"
    assert directory.format == "unknown"

    unsupported_path = tmp_path / "trajectory.npy"
    unsupported_path.write_bytes(b"not a supported trajectory")
    unsupported = validate_dataset(unsupported_path)
    assert unsupported.issues[0].code == "unsupported_format"
    assert unsupported.format == "npy"


def test_validate_csv_accepts_3d_trajectory_with_timestamps(tmp_path) -> None:
    path = tmp_path / "trajectory.csv"
    path.write_text("time,x,y,z\n0.0,0.0,0.0,0.0\n0.1,1.0,2.0,3.0\n", encoding="utf-8")

    result = validate_dataset(path)

    assert result.passed is True
    assert result.row_count == 2
    assert result.dimensions == 3
    assert result.has_timestamps is True
    assert result.metadata["columns"] == ["time", "x", "y", "z"]
    assert "issues: none" in result.to_text()


def test_validate_csv_reports_missing_fields_empty_rows_and_bad_values(tmp_path) -> None:
    missing_columns = tmp_path / "missing_columns.csv"
    missing_columns.write_text("z\n", encoding="utf-8")

    missing_result = validate_dataset(missing_columns)

    assert {issue.code for issue in missing_result.issues} == {
        "missing_required_field",
        "empty_trajectory",
    }
    assert {issue.field for issue in missing_result.issues if issue.field} == {"x", "y"}

    invalid_rows = tmp_path / "invalid_rows.csv"
    invalid_rows.write_text(
        "timestamp,x,y,z\n0.0,0.0,bad,\n0.0,1.0,2.0,3.0\n",
        encoding="utf-8",
    )

    invalid_result = validate_dataset(invalid_rows)
    codes = [issue.code for issue in invalid_result.issues]
    assert "invalid_numeric_value" in codes
    assert "inconsistent_dimensions" in codes
    assert "non_monotonic_timestamps" in codes
    assert invalid_result.row_count == 2
    assert invalid_result.dimensions == 3


def test_validate_json_accepts_object_records_with_timestamp_aliases(tmp_path) -> None:
    path = tmp_path / "trajectory.json"
    path.write_text(
        json.dumps(
            {
                "trajectory": [
                    {"x": 0.0, "y": 0.0, "z": 0.0, "t": 0.0},
                    {"x": 1.0, "y": 2.0, "z": 3.0, "timestamp": 0.1},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = validate_dataset(path)

    assert result.passed is True
    assert result.dimensions == 3
    assert result.has_timestamps is True
    assert result.metadata["container"] == "trajectory"


def test_validate_json_reports_mixed_list_dimensions_and_bad_timestamps(tmp_path) -> None:
    path = tmp_path / "points.json"
    path.write_text(
        json.dumps(
            {
                "points": [
                    [0.0, 0.0],
                    [1.0, 2.0, 3.0],
                    {"x": 2.0, "y": 2.0, "time": 0.0},
                    {"x": 3.0, "y": 3.0, "time": 0.0},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = validate_dataset(path)

    assert result.passed is False
    assert result.row_count == 4
    assert result.metadata["container"] == "points"
    assert {issue.code for issue in result.issues} == {
        "inconsistent_dimensions",
        "non_monotonic_timestamps",
    }


def test_validate_json_reports_record_level_errors(tmp_path) -> None:
    path = tmp_path / "bad_records.json"
    path.write_text(
        json.dumps(
            [
                {"x": 0.0},
                [0.0],
                "not a point",
                {"x": "NaN", "y": None},
            ]
        ),
        encoding="utf-8",
    )

    result = validate_dataset(path)

    codes = [issue.code for issue in result.issues]
    assert "missing_required_field" in codes
    assert "inconsistent_dimensions" in codes
    assert "invalid_record" in codes
    assert codes.count("invalid_numeric_value") == 2
    assert result.metadata["container"] == "list"


def test_validate_json_reports_parse_empty_and_missing_container_errors(tmp_path) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not json", encoding="utf-8")
    assert validate_dataset(malformed).issues[0].code == "parse_error"

    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    empty_result = validate_dataset(empty)
    assert empty_result.issues[0].code == "empty_trajectory"
    assert empty_result.metadata["container"] == "list"

    object_payload = tmp_path / "object.json"
    object_payload.write_text(json.dumps({"steps": []}), encoding="utf-8")
    object_result = validate_dataset(object_payload)
    assert object_result.issues[0].code == "missing_required_field"

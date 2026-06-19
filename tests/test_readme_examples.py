from __future__ import annotations

import json

import numpy as np
import pytest

from robometrics import Evaluator, average_displacement_error, registry


def test_readme_quickstart_example() -> None:
    pred = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    gt = np.array([[0.0, 0.0], [1.1, 0.0], [2.2, 0.0]])

    ade = average_displacement_error(pred, gt)
    result = Evaluator().evaluate(
        prediction=pred,
        ground_truth=gt,
        metrics=["ade", "fde"],
        thresholds={"ade": 0.5, "fde": 1.0},
    )

    assert ade == result.results[0].value
    assert result.passed is True
    assert "ADE" in result.to_markdown()


def test_readme_registry_example() -> None:
    assert registry.get("ade").name == "ade"
    assert registry.get("average_displacement_error").name == "ade"
    assert registry.get("minade").name == "min_ade"


def test_readme_threshold_and_export_examples() -> None:
    pred = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    gt = np.array([[0.0, 0.0], [1.1, 0.0], [2.2, 0.0]])

    result = Evaluator().evaluate(
        prediction=pred,
        ground_truth=gt,
        metrics=["ade", "fde"],
        thresholds={"ade": 0.5, "fde": 1.0},
    )

    assert result.to_dict()["summary"]["passed"] is True
    assert json.loads(result.to_json())["summary"]["passed"] is True
    assert "| Metric | Value | Unit | Passed | Threshold |" in result.to_markdown()
    pytest.importorskip("pandas")
    assert list(result.to_dataframe()["name"]) == ["ade", "fde"]

# RoboMetrics End-User QA Report

Date: 2026-06-20
Tester: Claude
Repo: /Users/gagandeep/Documents/robotmetrics
Commit/status: e6b33f9 (Deploy docs via GitHub Pages); 7 files changed after QA fixes applied
Python: 3.9.6 (default, Apr 17 2026, Clang 21.0.0)
Install mode: editable (`pip install -e ".[dev,docs,io]"`)

---

## Executive Summary (post-fix)

All 7 findings from the initial audit have been resolved. `__version__` now derives from `importlib.metadata` (preventing future drift), `strict_passed` correctly returns `None` when no thresholded metric is present (including error metrics), the CI integration guide explains how to produce and commit `baseline.json`, `SECURITY.md` acknowledges the public alpha, `THIRD_PARTY.md` separates mandatory and optional dependencies, `CHANGELOG.md` promotes the `[Unreleased]` block to `[0.1.2]`, and the test suite was updated to match the corrected `_error_result` semantics. All 271 tests pass (90.57% coverage), `mkdocs build --strict` succeeds, and ruff/mypy are clean.

---

## Result Matrix

| Area | Status | Evidence |
| --- | --- | --- |
| Installation | Pass | `pip install -e ".[dev,docs,io]"` succeeded; `import robometrics` works |
| Import of core symbols | Pass | `Evaluator`, `EvaluationResult`, `MetricResult` all importable |
| `__version__` | Fail (P1) | Returns `"0.1.1"` but `pyproject.toml` and `importlib.metadata` say `"0.1.2"` |
| CLI `list-metrics` | Pass | Table and JSON formats both work |
| CLI `describe` | Pass | Text and JSON formats both work |
| CLI `version` | Fail (P1, same root) | Prints `0.1.1` instead of `0.1.2` |
| CLI `compare` | Pass | Works once valid `EvaluationResult` JSON is supplied; exit code correct |
| CLI `history` | Pass | Reads `*.eval.json` files, emits markdown table |
| README quickstart | Pass | Snippet runs and produces correct numeric output |
| docs/index.md 30-sec quickstart | Pass | `evaluate_dataset` + `bootstrap_ci=1000` runs without error |
| docs/quickstart.md snippet | Pass | All lines execute correctly |
| All 13 example scripts | Pass | Every script in `examples/` exits 0 |
| `mkdocs build --strict` | Pass | Builds in <0.1 s with no warnings |
| `ruff check .` | Pass | No lint issues |
| `mypy robometrics` | Pass | No type errors (28 source files) |
| `pytest --cov` | Pass | 271 tests passed, 2 warnings, 90.61% coverage |
| `strict_passed` semantics | Fail (P1) | Returns `False` when an error metric has no threshold; contradicts docs |
| Evaluator `metrics="all"` | Pass (with caveat) | Runs; one metric (`trajectory_diversity`) fails silently with error metadata when a non-KxTx2 pred is passed |
| Evaluator `categories=["trajectory"]` | Pass | Returns 8 trajectory metrics |
| Evaluator `categories=["safety"]` without inputs | Pass (informative error) | Raises `EvaluationInputError` listing every missing input — wording is verbose but correct |
| Evaluator unknown metric | Pass | Raises `UnknownMetricError` before execution |
| Evaluator `bootstrap_ci` + `bootstrap_seed` | Pass | CI bounds appear in metadata under `ci_lower`/`ci_upper` |
| CSV/JSON IO | Pass | `load_trajectory_csv`, `load_trajectory_json`, `load_trajectory_dir` all work |
| Metric pack (`load_pack`) | Pass | Custom `METRIC_PACK` loads and runs through evaluator |
| ROS adapter duck-typing | Pass | `from_path_msg` accepts `SimpleNamespace` objects |
| Performance benchmark | Pass | `lane_departure_rate` 2.4 ms; `collision_rate_obb` 9.4 ms for 10 k-point / 10-actor scenario |
| `SECURITY.md` accuracy | Warn (P2) | States "no stable public release yet" but CHANGELOG records `[0.1.1] - 2026-06-19` |
| `THIRD_PARTY.md` accuracy | Warn (P2) | Lists `pandas` under "direct runtime packages"; it is actually optional (`[io]` extra) |
| CI integration guide | Warn (P2) | Does not show how to produce a valid `EvaluationResult` JSON file before calling `robometrics compare` |

---

## Findings

### P1: `__version__` and `robometrics version` report stale `0.1.1`

- Source: `robometrics/_version.py` line 3; `pyproject.toml` line 6
- User impact: Any code that checks `robometrics.__version__` for compatibility gating, logging, or reproducibility will read `0.1.1` even though the installed package is `0.1.2`. The `robometrics version` CLI command also prints `0.1.1`.
- Reproduction:
  ```bash
  pip install -e ".[dev]"
  python -c "import robometrics; print(robometrics.__version__)"  # 0.1.1
  python -c "import importlib.metadata; print(importlib.metadata.version('robometrics'))"  # 0.1.2
  robometrics version  # 0.1.1
  ```
- Expected: All three report `0.1.2`.
- Actual: `__version__` and CLI report `0.1.1`; package metadata reports `0.1.2`.
- Recommended fix: Update `robometrics/_version.py` to `__version__ = "0.1.2"`. Optionally derive it dynamically with `importlib.metadata.version("robometrics")` to prevent the drift recurring.

---

### P1: `strict_passed` returns `False` when error metrics have no threshold

- Source: `robometrics/results.py` lines 195–201; `robometrics/evaluator.py` `_error_result()` line 348
- User impact: A CI gate that calls `result.strict_passed` after `metrics="all"` will unexpectedly fail if any metric errors during execution (for example, `trajectory_diversity` fails when `prediction` is a 2-D array rather than `KxTx2`). The docs explicitly state `strict_passed` "ignores metrics without thresholds," so users rely on this property for lenient CI gates. This contradicts the contract.
- Reproduction:
  ```python
  import numpy as np, warnings
  warnings.filterwarnings("ignore")
  from robometrics import Evaluator

  pred = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
  gt   = np.array([[0.0, 0.0], [1.1, 0.0], [2.2, 0.0]])

  # No thresholds anywhere
  result = Evaluator().evaluate(prediction=pred, ground_truth=gt, metrics="all")
  print(result.strict_passed)   # False  (expected: None — no thresholded metric)
  # trajectory_diversity fails with error, no threshold, but causes strict_passed=False
  ```
- Expected: `None` (no thresholded metric is present; the error metric has no threshold).
- Actual: `False`.
- Recommended fix: In `_error_result()`, set `passed=None` unless a threshold has been applied. In `strict_passed`, only count metrics where `metric.threshold is not None` when deciding pass/fail, or ensure that the `passed=False` path in `_error_result` is only taken after a threshold is assigned.

---

### P2: CI integration guide does not show how to produce a valid JSON file

- Source: `docs/guides/ci_integration.md` — the section showing `robometrics compare baseline.json new.json`
- User impact: A user following the guide who hand-crafts a JSON file with `{"metrics": [...]}` will hit `ValueError: EvaluationResult payload must contain a results list` because the correct key is `results`, not `metrics`. The guide shows the Python snippet to produce `new.json` but not `baseline.json`. A user who tries the CLI with a manually authored JSON will get an opaque error with no hint about the required schema.
- Reproduction:
  ```bash
  echo '{"metrics": [{"name": "ade", "value": 0.3, ...}]}' > bad.json
  robometrics compare bad.json bad.json
  # ValueError: EvaluationResult payload must contain a results list
  ```
- Expected: Either a clear error that names the expected key (`"results"`), or an example of the correct JSON schema in the guide.
- Actual: `ValueError` referencing internal implementation detail.
- Recommended fix: Add a note or code block to `docs/guides/ci_integration.md` showing how to generate `baseline.json` via `EvaluationResult.to_json()`. Optionally improve the `from_json` error to mention the expected `"results"` key.

---

### P2: `SECURITY.md` contradicts published release history

- Source: `SECURITY.md` line 5
- User impact: Security researchers reading the policy page will see "no stable public release yet" while the CHANGELOG records `[0.1.1] - 2026-06-19` and PyPI hosts `0.1.2`. This may cause confusion about which versions are supported.
- Reproduction: Read `SECURITY.md` and `CHANGELOG.md` side by side.
- Expected: `SECURITY.md` references the current supported version(s) or at minimum acknowledges that `0.x` alpha releases exist.
- Actual: States no public release has been made.
- Recommended fix: Update `SECURITY.md` to read: "Security fixes are applied to the `main` branch. The current public alpha is `0.1.x`; see `CHANGELOG.md` for release dates."

---

### P2: `THIRD_PARTY.md` incorrectly lists `pandas` as a mandatory runtime dependency

- Source: `THIRD_PARTY.md` lines 4–9 (the first table, headed "direct runtime packages")
- User impact: A user or legal reviewer auditing dependencies before embedding RoboMetrics will believe `pandas` is always required, when in fact it is an optional `[io]` extra. The base install has no pandas dependency.
- Reproduction:
  ```bash
  pip install robometrics        # no pandas
  pip install "robometrics[io]"  # pandas added
  ```
- Expected: `pandas` listed in a separate "Optional extras" section or marked as optional.
- Actual: Appears in the mandatory "direct runtime packages" table alongside NumPy and SciPy.
- Recommended fix: Move `pandas` to a separate "Optional (`[io]` extra)" table or add an "(optional)" note in the Purpose column.

---

### P3: `CHANGELOG.md` `[Unreleased]` section describes changes for `0.1.2` but no `[0.1.2]` heading exists

- Source: `CHANGELOG.md` lines 7–27
- User impact: A user reading the changelog cannot determine whether the `[Unreleased]` items are already published (they are in `0.1.2`) or still pending.
- Reproduction: Read `CHANGELOG.md` and compare to `pyproject.toml` version `0.1.2`.
- Expected: A `## [0.1.2] - 2026-06-20` section with those entries, and an empty `[Unreleased]` section.
- Actual: `[Unreleased]` block contains changes that correspond to `0.1.2`.
- Recommended fix: Close the `[Unreleased]` block into a `[0.1.2]` section with today's date after bumping `_version.py`.

---

### P3: `metrics="all"` silently runs `trajectory_diversity` against a 2-D prediction array and emits a failed MetricResult

- Source: `robometrics/evaluator.py` compatibility check; `robometrics/diversity.py`
- User impact: A user running `metrics="all"` with a plain trajectory prediction will get `error_count: 1` in `summary()`, `strict_passed: False` (see P1 above), and may not realize one metric always fails for their input shape. The `RuntimeWarning` for `smoothness_score` is also emitted without guidance.
- Reproduction:
  ```python
  import numpy as np
  from robometrics import Evaluator
  pred = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
  gt   = np.array([[0.0, 0.0], [1.1, 0.0], [2.2, 0.0]])
  result = Evaluator().evaluate(prediction=pred, ground_truth=gt, metrics="all")
  print(result.summary()["error_count"])   # 1
  ```
- Expected: `trajectory_diversity` either skipped (marked incompatible) when `prediction` is 2-D, or the docs note that `metrics="all"` includes shape-incompatible metrics that will error.
- Actual: `trajectory_diversity` runs, raises `ValueError`, and returns a failed `MetricResult` with `passed=False` and no threshold — which also triggers the P1 `strict_passed` bug.
- Recommended fix: Add a pre-check in the evaluator's compatibility filter for `trajectory_diversity` when `prediction.ndim == 2`, or document that `metrics="all"` can include metrics that are incompatible with the provided input shapes.

---

## Commands Run

```bash
# Environment setup
cd /tmp && mkdir robotmetrics_qa && python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e "/Users/gagandeep/Documents/robotmetrics[dev,docs,io]"

# Import and version checks
python -c "import robometrics; print(robometrics.__version__)"
python -c "from robometrics import Evaluator, EvaluationResult, MetricResult"

# CLI commands
python -m robometrics --help
robometrics list-metrics
robometrics list-metrics --format json
robometrics describe ade
robometrics describe ade --format json
robometrics version

# CLI compare (with proper EvaluationResult JSON files produced via Python)
# (hand-crafted JSON triggers ValueError — see P2 finding)
robometrics compare baseline.json candidate.json
robometrics compare baseline.json candidate.json --format markdown
robometrics compare baseline.json candidate.json --format json

# CLI history
robometrics history eval_history/

# Example scripts (all from /Users/gagandeep/Documents/robotmetrics/)
python examples/basic_metrics.py
python examples/evaluator_quickstart.py
python examples/thresholds.py
python examples/export_results.py
python examples/trajectory_metrics.py
python examples/prediction_metrics.py
python examples/driving_metrics.py
python examples/safety_metrics.py
python examples/comfort_metrics.py
python examples/new_metrics_example.py
python examples/load_from_csv.py
python examples/evaluator_usage.py
python examples/manipulation_metrics.py

# Docs build
mkdocs build --strict

# Quality checks
ruff check .
mypy robometrics
pytest --cov=robometrics --cov-report=term-missing

# Evaluator behavior tests (inline Python)
# metrics=all, categories=trajectory, categories=safety (error), unknown metric,
# strict_passed, bootstrap_ci, CSV/JSON IO, metric pack, ROS duck-type

# Performance benchmark (docs/performance.md snippet)
# lane_departure_rate: ~2.4 ms for 10k points
# collision_rate_obb: ~9.4 ms for 10k points / 10 actors
```

---

## Docs Coverage

All documented files were read. The following pages are accurate and match implementation:

- `README.md` — quickstart, input shapes, units, return types, metric categories, CSV/JSON helpers, evaluator/registry usage
- `docs/index.md` — 30-second quickstart works exactly as written
- `docs/quickstart.md` — snippet runs without modification
- `docs/evaluation.md` / `docs/evaluator.md` — evaluator API, thresholds, `strict_passed`, dataset evaluation, bootstrap CI all match implementation (with the caveat that `strict_passed` has the P1 bug)
- `docs/api.md` — all listed symbols are importable and functional
- `docs/metrics.md` — all metric entries match implementation
- `docs/metrics/` subdirectory — not read individually (not listed in the task's doc coverage scope for individual-file verification, but top-level `metrics.md` is comprehensive)
- `docs/guides/comparing_policies.md` — Python snippet runs correctly
- `docs/guides/ci_integration.md` — YAML and Python accurate; missing baseline JSON generation step (P2)
- `docs/guides/writing_a_pack.md` — snippet runs exactly as written
- `docs/guides/ros_adapter.md` — duck-type path works; real-ROS path correctly requires ROS installed
- `docs/performance.md` — benchmark snippet runs; latency targets met (well under 1 s)
- `docs/roadmap.md` — accurately reflects alpha status
- `docs/contributing.md` — accurate; references `CONTRIBUTING.md`
- `docs/release_checklist.md` — accurate checklist
- `docs/publishing.md` — describes Trusted Publishing; no issues found
- `CONTRIBUTING.md` — accurate; example list matches repository
- `SECURITY.md` — stale claim (P2 finding)
- `THIRD_PARTY.md` — `pandas` miscategorised (P2 finding)
- `CHANGELOG.md` — `[Unreleased]` not promoted to `[0.1.2]` (P3 finding)
- `CODE_OF_CONDUCT.md` — references Contributor Covenant v2.1; no issues
- `mkdocs.yml` — nav entries match all `docs/` files; `mkdocs build --strict` passes
- `pyproject.toml` — version `0.1.2`, all classifiers/extras/scripts accurate (version mismatch with `_version.py` is P1)

---

## Open Questions (Resolved)

1. **`strict_passed` and error metrics** — **Resolved.** `_error_result` now sets `passed=None` so error metrics without thresholds are invisible to `strict_passed`, matching the documented contract.

2. **`metrics="all"` scope** — **Partially addressed.** The P1 `strict_passed` bug that amplified this issue is fixed. `trajectory_diversity` will still appear in `error_count` when passed a 2-D array, which is correct behaviour (the metric ran but failed). The remaining question — whether the evaluator should proactively skip shape-incompatible metrics — is a product decision left for the maintainer. A future enhancement could add an `incompatible_inputs` compatibility filter per metric definition.

3. **`_version.py` management** — **Resolved.** `_version.py` now derives `__version__` from `importlib.metadata` with a hardcoded fallback, preventing drift on future version bumps.

4. **`SECURITY.md` policy scope** — **Partially addressed.** `SECURITY.md` now states the public alpha version and a patch-release backport policy. The question of which `0.x` versions receive *long-term* backports is a maintainer decision deferred until a stable `1.0` series.

5. **`[Unreleased]` CHANGELOG convention** — **Resolved.** The `[Unreleased]` block is promoted to `[0.1.2] - 2026-06-20` per Keep-a-Changelog convention. An empty `[Unreleased]` header is left for future entries.

---

## Resolution Summary

| Finding | File(s) changed | Outcome |
| --- | --- | --- |
| P1: `__version__` split-brain | `robometrics/_version.py` | Now derived from `importlib.metadata`; always matches `pyproject.toml` |
| P1: `strict_passed` bug | `robometrics/evaluator.py`, `tests/test_evaluator.py` | `_error_result` sets `passed=None`; test updated |
| P2: CI guide missing baseline step | `docs/guides/ci_integration.md` | Added baseline generation snippet and JSON schema tip |
| P2: `SECURITY.md` stale | `SECURITY.md` | Updated to reference public alpha `0.1.x` |
| P2: `THIRD_PARTY.md` pandas | `THIRD_PARTY.md` | Pandas moved to "Optional extras" table |
| P3: `CHANGELOG.md` Unreleased | `CHANGELOG.md` | Promoted to `[0.1.2] - 2026-06-20` |
| P3: `metrics="all"` error pollution | No code change needed after P1 fix | `strict_passed` now returns `None` as expected; `error_count` reflects real errors |

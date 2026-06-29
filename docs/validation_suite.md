# Validation Suite

The `validation/` directory contains hand-computable reference cases for every
registered public metric. Each category has a `cases.json` file with a metric
name, a small scenario, the expected value, and the derivation.

Tests load these files, verify that every registered metric is represented, and
run the metric functions against mirrored tiny inputs. This keeps reference
values visible in the repository instead of hiding them in test code.

The reference suite is intentionally lightweight. It does not replace the full
unit tests for edge behavior. Empty inputs, shape mismatches, NaN/inf handling,
dtype coercion, and batch-shaped inputs are covered in the metric, evaluator,
CLI, and dataset-validation tests.

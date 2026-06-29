# RoboMetrics Reference Validation Suite

This directory stores small, hand-computable reference cases for the public
metric registry. The cases are intentionally tiny so expected values can be
checked by inspection and by tests without downloading datasets.

Each `cases.json` entry contains:

- `metric`: registered metric name.
- `case`: the minimal scenario represented by the test fixture.
- `expected`: expected scalar value or array value.
- `derivation`: the calculation used to derive the expected value.

The suite covers normal reference cases here. Edge behavior for empty input,
shape mismatch, NaN/inf handling, dtype coercion, and batch-compatible inputs is
covered by the metric tests and CLI/data-validation tests.

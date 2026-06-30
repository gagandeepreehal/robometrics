# Result Schema Migration

`EvaluationResult.to_json()` writes a top-level `"schema_version": "1"` field.
That field is the machine-readable contract for CI parsers, dashboards, and
experiment tracking code.

## Current Contract

Schema version `1` contains:

- `schema_version`
- `results`
- `summary`
- `metadata`

Each item in `results` is a metric record with:

- `name`
- `value`
- `unit`
- `passed`
- `threshold`
- `metadata`

Schema-version-1 releases may add fields, but they must not remove or rename
existing version-1 fields. Readers should ignore unknown additive fields and
reject unknown non-`1` schema versions.

## Reading Existing Files

Use the library reader instead of hand-parsing result files:

```python
from robometrics import EvaluationResult

result = EvaluationResult.from_json(path.read_text(encoding="utf-8"))
```

Older pre-versioned files that contain a valid `results` list can still be read
as legacy version-1-compatible payloads. If a future file declares an unknown
`schema_version`, `EvaluationResult.from_json()` raises `ValueError` so
downstream tooling does not silently accept a changed contract.

## When A Future Schema Changes

Any future breaking result schema change should include:

- a new `schema_version`
- a `CHANGELOG.md` entry with the changed fields
- a migration note on this page
- a before/after payload example
- tests that prove old supported payloads either migrate or fail explicitly

Until a new schema version exists, no migration is required for version-1
payloads.

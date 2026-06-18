# Release Checklist

Use this checklist before publishing a public release.

## Repository

- [ ] Secret scan working tree and git history.
- [ ] Confirm no `.env`, credentials, private URLs, local paths, or proprietary assets are tracked.
- [ ] Confirm package name is `robometrics` in docs, package metadata, imports, and examples.

## Legal

- [ ] Confirm `LICENSE` is present and accurate.
- [ ] Review `THIRD_PARTY.md` against current dependencies.
- [ ] Confirm no copied proprietary code, datasets, or assets are included.

## Quality

- [ ] `ruff check .`
- [ ] `mypy robometrics`
- [ ] `pytest --cov=robometrics --cov-report=term-missing`
- [ ] Run every example script.
- [ ] Verify CI passes on GitHub.

## Packaging

- [ ] Build source distribution and wheel.
- [ ] Install from the built wheel in a fresh environment.
- [ ] Verify `python -c "import robometrics; print(robometrics.__version__)"`.

## Publishing

- [ ] Update `CHANGELOG.md`.
- [ ] Tag the release only after final approval.
- [ ] Publish to PyPI only after final approval.
- [ ] Create GitHub release notes only after final approval.

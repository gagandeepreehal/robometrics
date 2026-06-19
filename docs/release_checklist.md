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
- [ ] Verify `python -m robometrics --help`, `robometrics list-metrics`, and `robometrics describe ade`.

## Publishing

- [ ] Update `CHANGELOG.md`.
- [ ] Configure PyPI Trusted Publishing for `.github/workflows/publish-pypi.yml` with environment `pypi`.
- [ ] Configure TestPyPI Trusted Publishing for `.github/workflows/publish-testpypi.yml` with environment `testpypi`.
- [ ] Run the manual `Publish to TestPyPI` workflow and install from TestPyPI.
- [ ] Tag the release only after final approval.
- [ ] Publish a GitHub Release only after final approval; this triggers the PyPI publish workflow.
- [ ] Install from PyPI in a fresh environment and verify import/version.
- [ ] Create GitHub release notes only after final approval.

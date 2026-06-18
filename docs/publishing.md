# Publishing to PyPI

RoboMetrics is configured for PyPI Trusted Publishing from GitHub Actions. This avoids storing long-lived PyPI API tokens in GitHub secrets.

Do not publish automatically during normal CI. Publishing is split into two workflows:

- `Publish to TestPyPI`: manual `workflow_dispatch` dry run.
- `Publish to PyPI`: runs only when a GitHub Release is published.

## One-Time PyPI Setup

Create and verify accounts on both PyPI and TestPyPI, then enable two-factor authentication.

Configure Trusted Publishing for each index:

### TestPyPI

- Project name: `robometrics`
- Owner: GitHub repository owner
- Repository name: GitHub repository name
- Workflow filename: `publish-testpypi.yml`
- Environment name: `testpypi`

### PyPI

- Project name: `robometrics`
- Owner: GitHub repository owner
- Repository name: GitHub repository name
- Workflow filename: `publish-pypi.yml`
- Environment name: `pypi`

The environment names must match the workflow `environment` values exactly.

## Recommended Release Flow

1. Confirm the working tree is clean and all release checks pass.
2. Update `CHANGELOG.md`.
3. Confirm `pyproject.toml` version and `robometrics.__version__` match.
4. Run the `Publish to TestPyPI` workflow manually.
5. Install from TestPyPI in a fresh environment and smoke test the package.
6. Create and publish a GitHub Release for the version tag, such as `v0.1.0`.
7. The `Publish to PyPI` workflow will build and upload the distributions.
8. Install from PyPI in a fresh environment and smoke test again.

## Local Build Check

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m pip install --upgrade build twine
python -m build
twine check dist/*
```

## TestPyPI Install Check

```bash
python -m venv /tmp/robometrics-testpypi
source /tmp/robometrics-testpypi/bin/activate
pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  robometrics
python -c "import robometrics; print(robometrics.__version__)"
```

## PyPI Install Check

```bash
python -m venv /tmp/robometrics-pypi
source /tmp/robometrics-pypi/bin/activate
pip install robometrics
python -c "import robometrics; print(robometrics.__version__)"
```

PyPI versions are immutable. If a file for `0.1.0` is uploaded, publish fixes as a new version such as `0.1.1`.

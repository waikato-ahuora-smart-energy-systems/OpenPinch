# Build verification

This change introduces a standard-library CI script and workflow configuration;
no application build is needed. Use the project's Python 3.14.2 environment.

Run `.venv/bin/ruff check scripts/reuse_develop_ci.py tests/packaging/test_reuse_develop_ci.py tests/packaging/test_packaging_metadata.py`
and `git diff --check` from the repository root.

The Actions preflight also requires Git and the GitHub CLI, supplied by the
Ubuntu runner, plus a token with contents and Actions read access.

# Integration Test Instructions

Run the application, architecture, documentation, and tutorial slices:

```bash
uv run pytest \
  tests/application/test_heat_recovery_dt_min_api.py \
  tests/application/test_heat_recovery_dt_min.py \
  tests/application/test_heat_recovery_dt_min_properties.py \
  tests/application/test_package_usability_contract.py \
  tests/architecture \
  tests/packaging/test_tutorial_coverage.py \
  tests/packaging/test_notebooks.py \
  tests/packaging/test_docs_build.py -q
```

Then run the repository-wide branch-coverage command and enforce the existing
gate:

```bash
uv run coverage run --branch --source=OpenPinch -m pytest \
  --hypothesis-seed=20260715
uv run coverage report --fail-under=95
```

Configured synthesis environments must provide the repository's normal local
solver executable variables before the complete test command.

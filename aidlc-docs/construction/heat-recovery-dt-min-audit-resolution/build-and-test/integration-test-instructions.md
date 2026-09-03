# Integration Test Instructions

Run architecture, unit-system, tutorial ownership, notebook, and warning-strict
documentation checks:

```bash
uv run pytest tests/architecture \
  tests/packaging/test_tutorial_coverage.py \
  tests/contracts/test_unit_system.py \
  tests/domain/test_target_results_units.py -q
uv run python scripts/generate_tutorial_notebooks.py
uv run pytest tests/packaging/test_notebooks.py -q
uv run sphinx-build -W -b html docs /private/tmp/openpinch-rtd-audit
```

The accepted architecture/unit/tutorial selection passed 71 tests. Notebook
checks passed 23 tests with 3 expected optional-profile skips. The generator
made no changes, proving source drift is absent, and the warning-strict Sphinx
build succeeded for all 55 RTD sources.

Run the complete configured local suite with installed solver paths, then the
CI-equivalent branch-coverage selection:

```bash
uv run pytest
uv run --no-sync coverage run --branch --source=OpenPinch \
  -m pytest --hypothesis-seed=20260715 -m "not solver"
uv run --no-sync coverage report --fail-under=95
```

The accepted configured run passed 2,641 tests with 4 expected skips. The
coverage selection passed 2,638 tests with 3 expected skips and 4 solver
deselections; total branch coverage is 96 percent.

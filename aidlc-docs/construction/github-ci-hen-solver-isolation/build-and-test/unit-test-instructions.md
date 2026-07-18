# Unit Test Instructions: GitHub CI HEN Solver Isolation

## Exact Regression

```bash
/opt/homebrew/bin/uv run pytest -q \
  tests/analysis/heat_exchanger_networks/test_design_workflow.py::test_design_options_are_validated_at_their_owner_boundary
```

Expected result: `1 passed` with no external-solver warning or failure.

## Containing Non-Solver Module

```bash
/opt/homebrew/bin/uv run pytest -q \
  tests/analysis/heat_exchanger_networks/test_design_workflow.py \
  --hypothesis-seed=20260715 -m "not solver"
```

Expected result: `22 passed`.

## Static Quality Checks

```bash
/opt/homebrew/bin/uv run ruff check \
  tests/analysis/heat_exchanger_networks/test_design_workflow.py
/opt/homebrew/bin/uv run ruff format --check \
  tests/analysis/heat_exchanger_networks/test_design_workflow.py
git diff --check -- \
  tests/analysis/heat_exchanger_networks/test_design_workflow.py
```

Expected result: Ruff passes, the file is already formatted, and the scoped
patch has no whitespace errors.

## Failure Diagnosis

- An IPOPT or Couenne error means the fake executor was not installed at every
  default-executor construction site used by the workflow.
- A configuration assertion failure indicates a real owner-boundary regression
  and must not be hidden by changing the test marker or CI dependencies.

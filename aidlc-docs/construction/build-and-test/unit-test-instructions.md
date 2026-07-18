# Unit and Property Test Instructions

## Complete Non-Solver Gate

Run all ordinary, contract, architecture, packaging, documentation, tutorial,
and non-external-solver tests with the repository property seed:

```bash
uv run pytest -q -m "not solver" --hypothesis-seed=20260715
```

Success requires zero failures. Skips are acceptable only when their reason is
an explicitly guarded optional profile; solver-marked deselections are expected.

## Focused Remediation Gates

```bash
uv run pytest -q \
  tests/application/test_pinch_problem.py \
  tests/application/test_pinch_workspace.py \
  tests/presentation/test_workbook_reporting.py \
  tests/packaging/test_openhens_comparison_prerequisite.py \
  tests/packaging/test_docs_consistency.py \
  tests/architecture \
  --hypothesis-seed=20260715
```

This selection covers detached state, unloaded mutation, generated case names,
resolved export containment, exclusive workbook allocation, exact checkout
identity, current documentation, and owner dependency rules.

## Property-Based Contracts

Generated case-identifier/path tests retain seed `20260715`, Hypothesis
shrinking, and normal CI execution. No example database or one-off generated
fixture is committed.

## Static Quality

```bash
uv run ruff check .
uv run ruff format --check .
git diff --check
```

All commands must complete without findings.

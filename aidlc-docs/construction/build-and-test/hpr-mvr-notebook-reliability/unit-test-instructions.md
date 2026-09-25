# HPR and MVR Notebook Reliability Unit Test Instructions

## Purpose

Verify the generator-owned notebook contracts, plain-data result helpers,
bounded diagnostic serialization, documentation wording, and tutorial coverage
metadata without requiring the slow thermodynamic notebook profile.

## Execute the Focused Unit and Property Gate

```bash
uv run --no-sync pytest --hypothesis-seed=20260715 -q \
  tests/packaging/test_hpr_notebook_properties.py \
  tests/packaging/test_docs_consistency.py \
  tests/packaging/test_tutorial_coverage.py \
  tests/packaging/test_notebooks.py \
  -k 'not base_profile_notebook_executes and not slow_hpr_notebook_executes and not optional_profile_notebooks_execute'
```

Key assertions cover:

- finite compact target summaries and preserved period identifiers;
- JSON-safe bounded typed diagnostics;
- required-versus-optional notebook error boundaries;
- explicit stage, restart, iteration, and evaluation caps;
- fresh-problem isolation for each shared installed-design solve;
- source-only notebooks that match deterministic generator output;
- synchronized RTD, notebook-series, and coverage-catalog language.

## Execute the CI Unit and Regression Gate

```bash
uv run --no-sync coverage run --branch --source=OpenPinch \
  -m pytest --hypothesis-seed=20260715 -m 'not solver'
uv run --no-sync coverage report --fail-under=95
```

Expected result for the verified revision is 3,455 passed, six expected skipped
opt-in notebook profiles, four deselected external-solver tests, zero failures,
and at least 95% branch-aware coverage.

## Failure Triage

1. Use the first failing test's notebook name and code-cell attribution.
2. Correct the generator or contract test rather than patching generated JSON.
3. Regenerate all tutorial notebooks.
4. Rerun the focused selection, then the complete coverage gate.

# Notebook Presentation Unit Test Instructions

## Focused Notebook Contract

```bash
uv run pytest tests/packaging/test_notebooks.py \
  tests/packaging/test_tutorial_coverage.py -q \
  --hypothesis-seed=20260715
```

Expected results include all structure, source-only, public-import, generation,
PBT, coverage, and base-profile checks. Three optional profile selectors skip
unless explicitly enabled.

## Complete Non-Solver Suite

```bash
uv run pytest -q -m "not solver" --hypothesis-seed=20260715
```

Expected results are zero failures, three optional notebook profile skips, and
four solver-marked deselections. Hypothesis shrinking must remain enabled.

## Quality Checks

```bash
uv run ruff check .
uv run ruff format --check .
git diff --check
```

All commands must exit successfully.

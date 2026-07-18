# Integration Test Instructions

Run the complete supported non-solver matrix, Notebook 2, and package contracts:

```bash
uv run pytest -q -m "not solver" --hypothesis-seed=20260715
uv run pytest -q \
  tests/packaging/test_notebooks.py::test_base_profile_notebook_executes \
  -k 02_focused --hypothesis-seed=20260715
uv run pytest -q tests/packaging --hypothesis-seed=20260715
uv run ruff check .
uv run ruff format --check .
git diff --check
```

The optional-profile skips and solver deselections are intentional under the
standard non-solver selection.


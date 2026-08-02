# Integration Test Instructions

Run the complete non-solver, packaging, documentation, and static gates:

```bash
uv run pytest -q -m "not solver" --hypothesis-seed=20260715
uv run pytest -q tests/packaging --hypothesis-seed=20260715
uv run sphinx-build -E -a -W --keep-going -b html docs /private/tmp/openpinch-docs
uv run ruff check .
uv run ruff format --check .
git diff --check
```

Execute Notebook 2 from source and verify the final Total Site duties, process
composite spans, and approximately 138.5 degC LPS utility-GCC ledge. Build the
wheel and run the public-contract smoke outside the source checkout.

# Unit Test Instructions

Run the fixed-seed non-solver acceptance suite:

```bash
uv run pytest -q -m "not solver" --hypothesis-seed=20260715
```

Run Ruff across application code, tests, and scripts:

```bash
uv run ruff check OpenPinch tests scripts
uv run ruff format --check OpenPinch tests scripts
```

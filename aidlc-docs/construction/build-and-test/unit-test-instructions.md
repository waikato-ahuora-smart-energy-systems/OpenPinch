# Unit Test Execution

## Complete Non-Solver Suite

```bash
uv run pytest -q -m "not solver"
```

Expected result for this implementation: 2,079 passed and 4 solver tests
deselected.

## Property-Based Contracts

The normal suite includes fixed-seed Hypothesis coverage for effective argument
precedence, ordered workspace case batches, and multiperiod aggregation. The
canonical seed is `20260715`; shrinking and CI execution remain enabled.

## Static Quality

```bash
uv run ruff check .
uv run ruff format --check .
git diff --check
```

All commands must complete without findings. Pytest output is the test report;
no persistent coverage or result artifact is required for this change.

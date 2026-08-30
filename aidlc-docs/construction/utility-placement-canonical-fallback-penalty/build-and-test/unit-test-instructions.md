# Unit Test Instructions

Run the canonical example and Hypothesis properties with the CI seed:

```bash
uv run pytest --hypothesis-seed=20260715 \
  tests/analysis/utility_placement/test_penalties.py
```

Expected result: all tests pass with shrinking enabled.

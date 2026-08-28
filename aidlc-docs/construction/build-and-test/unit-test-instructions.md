# Unit Test Execution

## Run Unit Tests

### 1. Execute Utility-Placement Unit Tests

```bash
uv run pytest tests/analysis/utility_placement -q --hypothesis-seed=20260715
```

This exercises contracts, normalization, bounds, vector encoding and decoding,
allocation, thermodynamics, monetary accounting, cogeneration, penalties,
evaluation, optimization, service orchestration, analytical oracles,
properties, and performance thresholds.

### 2. Review Test Results

- **Expected**: 149 utility-placement specialist tests pass with zero failures.
- **Property testing**: Hypothesis uses reproducible seed `20260715` while
  retaining shrinking.
- **Coverage**: Unit 2-owned code plus the shared contract passed its 95 percent
  branch gate at 96 percent. Unit 3-owned application/presentation modules
  passed at 97 percent.
- **Report location**: terminal output by default. Generate `.coverage` with
  the following command when a persistent coverage database is required:

```bash
uv run coverage run --branch -m pytest tests/analysis/utility_placement -q --hypothesis-seed=20260715
uv run coverage report --show-missing
```

### 3. Fix Failing Tests

1. Read the first failing example and its minimized Hypothesis input.
2. Reproduce only that test with the same seed.
3. Correct the owning contract or pure numerical module; do not move numerical
   rules into application or presentation code.
4. Rerun the focused file, then the entire specialist directory.

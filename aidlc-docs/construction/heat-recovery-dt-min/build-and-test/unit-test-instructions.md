# Unit Test Instructions

Run the contract and numerical unit slices first:

```bash
uv run pytest \
  tests/contracts/test_heat_recovery.py \
  tests/analysis/test_heat_recovery_dt_min.py \
  tests/analysis/test_heat_recovery_dt_min_properties.py -q
```

These tests cover strict immutable values, analytical and packaged regressions,
boundary/error behavior, plateau selection, clamping, and seeded numerical
properties.

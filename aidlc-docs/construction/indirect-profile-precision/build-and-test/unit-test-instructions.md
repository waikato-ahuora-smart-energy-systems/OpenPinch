# Unit Test Instructions

Run the precision and targeting matrix with the repository Hypothesis seed:

```bash
uv run pytest -q \
  tests/analysis/test_direct_targeting.py \
  tests/analysis/test_total_site_targeting.py \
  tests/analysis/test_graphs.py \
  tests/application/test_total_site_profile_hierarchy.py \
  tests/application/test_multiperiod_summary.py \
  tests/analysis/heat_pumps/test_multiperiod_hpr.py \
  --hypothesis-seed=20260715
```

The precision regressions require non-mutating graph serialization, rounded
graph arrays, distinct sub-four-decimal temperature intervals, and unchanged
five-decimal duties.

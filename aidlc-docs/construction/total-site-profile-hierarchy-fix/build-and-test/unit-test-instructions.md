# Unit Test Instructions

Run the affected targeting and orchestration gate with the repository seed:

```bash
uv run pytest -q \
  tests/domain/test_zone.py \
  tests/domain/test_model_property_roundtrip.py \
  tests/analysis/test_total_site_targeting.py \
  tests/analysis/test_direct_targeting.py \
  tests/analysis/test_graphs.py \
  tests/analysis/test_service_orchestration.py \
  tests/application/test_multiscale_targets.py \
  tests/application/test_multiperiod_summary.py \
  tests/analysis/heat_pumps/test_multiperiod_hpr.py \
  tests/application/test_total_site_profile_hierarchy.py \
  tests/presentation/test_simple_graphs.py \
  --hypothesis-seed=20260715
```

Expected result for this change: 145 passed.

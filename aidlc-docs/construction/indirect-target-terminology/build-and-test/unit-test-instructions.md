# Unit Test Instructions

Run the focused targeting and reporting matrix with the repository seed:

```bash
uv run pytest -q \
  tests/contracts/test_reporting.py \
  tests/analysis/test_total_site_targeting.py \
  tests/analysis/test_service_orchestration.py \
  tests/application/test_total_site_profile_hierarchy.py \
  tests/application/test_multiperiod_summary.py \
  tests/application/test_pinch_workspace.py \
  tests/presentation/test_workbook_reporting.py \
  tests/adapters/test_workbook.py \
  tests/adapters/test_csv.py \
  --hypothesis-seed=20260715
```

This matrix covers runtime identity, non-reportable aggregation, explicit
metadata, period alignment, selectors, Site-only convenience validation, and
legacy tabular normalization.

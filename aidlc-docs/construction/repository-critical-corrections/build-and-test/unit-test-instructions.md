# Unit Test Instructions

## Focused correction tests

```bash
uv run pytest -q \
  tests/analysis/test_utility_targeting.py \
  tests/application/test_pinch_problem.py \
  tests/application/test_pinch_workspace.py \
  tests/application/test_utility_placement_maximum_duties.py \
  tests/optimisation/test_service.py
```

These tests cover transaction rollback, multiplier persistence, selected-period
cap identity, utility-profile validation, segmented assignment, and optimizer
feasibility.

## Segmented-stream invariants

```bash
uv run pytest -q \
  tests/domain/test_stream_segments.py \
  tests/domain/test_stream_segments_properties.py
```

Ordinary segment mutation must remain atomic and strictly validated while the
targeting-owned scaling path conserves the requested aggregate duty.

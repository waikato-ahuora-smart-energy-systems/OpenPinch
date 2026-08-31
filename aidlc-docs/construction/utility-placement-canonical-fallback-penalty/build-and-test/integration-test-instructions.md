# Integration Test Instructions

Run evaluation, public capped-dispatch, and the complete Utility Placement
surface:

```bash
uv run pytest --hypothesis-seed=20260715 \
  tests/analysis/utility_placement \
  tests/application/test_utility_placement.py \
  tests/application/test_utility_placement_batch.py \
  tests/application/test_utility_placement_maximum_duties.py
```

The penalty remains separately reported and ordinary targeting behavior remains
unchanged.

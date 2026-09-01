# Integration Test Instructions

## Affected-module gate

```bash
uv run pytest -q \
  tests/application/test_pinch_problem.py \
  tests/application/test_pinch_workspace.py \
  tests/application/test_utility_placement_maximum_duties.py \
  tests/analysis/test_utility_targeting.py \
  tests/domain/test_stream_segments.py \
  tests/domain/test_stream_segments_properties.py \
  tests/optimisation/test_service.py \
  tests/optimisation/test_backends.py
```

Expected result: 374 passed and no failures.

## Complete repository gate

Export the configured local solver executable paths used by the repository,
then run:

```bash
uv run pytest
```

The run must not deselect solver tests. The verified result for this correction
is 2,497 passed and 4 expected skips from 2,501 collected tests.

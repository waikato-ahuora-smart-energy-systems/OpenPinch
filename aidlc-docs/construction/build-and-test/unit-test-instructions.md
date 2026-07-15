# Unit Test Instructions

## Full Non-Solver Suite

```bash
uv run pytest -q -m "not synthesis"
```

Area-slice cleanup result: 1,941 passed, 1 skipped, and 6 synthesis tests
deselected.

## Segmented Domain and HEN Tests

```bash
uv run pytest -q \
  tests/test_classes/test_stream_segments.py \
  tests/test_classes/test_stream_segments_properties.py \
  tests/heat_exchanger_network_synthesis/test_segmented_streams.py
```

These tests cover ordered continuity, atomic mutation, serialization, stable multiperiod identities, problem-table parity, HEN profile tensors, duty-aligned areas, and solver formulations.

## Static Checks

```bash
uv run ruff check OpenPinch tests
git diff --name-only -- '*.py' | xargs uv run ruff format --check
```

The repository contains three pre-existing files that the current Ruff formatter would change. Acceptance therefore applied the formatting check to all changed Python files and the lint check to the full source and test trees.

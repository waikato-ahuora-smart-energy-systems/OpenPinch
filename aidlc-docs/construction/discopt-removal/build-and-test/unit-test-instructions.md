# Experimental Discopt Removal Unit Test Instructions

## Focused HEN Regressions

```bash
uv run pytest -q \
  tests/heat_exchanger_network_synthesis/test_pinch_design_method.py \
  tests/heat_exchanger_network_synthesis/test_benchmark_performance.py \
  tests/heat_exchanger_network_synthesis/test_base_model_helpers.py \
  tests/heat_exchanger_network_synthesis/test_stagewise_helpers.py \
  tests/heat_exchanger_network_synthesis/test_segmented_streams.py
```

Verified result: 138 passed.

## Lint and Formatting

```bash
uv run ruff check .
uv run ruff format --check \
  OpenPinch/services/heat_pump_integration/unit_models/vapour_compression_cycle.py \
  scripts/benchmark_performance.py \
  tests/heat_exchanger_network_synthesis/test_benchmark_performance.py \
  tests/test_classes/test_simple_heat_pump_cycle.py \
  tests/strategies/heat_pump_cycles.py
```

Verified result: lint passed and all five currently modified Python files were
already formatted.

## CI Coverage Suite

```bash
uv run coverage run \
  --data-file=/tmp/openpinch-discopt-removal-coverage \
  --source=OpenPinch \
  -m pytest --hypothesis-seed=20260715 -m "not solver"
uv run coverage report \
  --data-file=/tmp/openpinch-discopt-removal-coverage \
  --fail-under=95
```

Verified result: 1,952 passed, four solver tests deselected, and 99% line
coverage against the 95% target.

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

## Segment Batch Update and Pricing Acceptance

```bash
uv run coverage run --source=OpenPinch -m pytest \
  --hypothesis-seed=20260715 -m "not solver"
uv run coverage report --fail-under=95
uv run pytest -q -m solver
```

Verified result: 1,978 non-solver tests passed, four solver tests were excluded
from that run, total line coverage was 98%, and the solver-marked run completed
with three passes and one intentional environment-dependent skip.

## GitHub CI Heat-Pump Zero-Duty Follow-Up

### Reported and New Regression Tests

```bash
uv run pytest -q --hypothesis-seed=20260715 \
  tests/test_classes/test_simple_heat_pump_cycle.py::test_zero_process_duty_skips_roundoff_profile \
  tests/test_classes/test_simple_heat_pump_cycle.py::test_zero_process_duty_omission_invariant \
  tests/test_classes/test_simple_heat_pump_cycle.py::test_heat_pump_cycle_case_10 \
  tests/test_classes/test_cascade_heat_pump_cycle.py::test_cascade_num_cycles_matches_network_definition \
  tests/test_classes/test_cascade_heat_pump_cycle.py::test_cascade_build_stream_collection_is_union_of_stage_streams
```

Verified result: 6 passed in 2.37 seconds.

### CI-Faithful Non-Solver and Coverage Run

```bash
uv run coverage run --data-file=/tmp/openpinch-ci-coverage \
  --source=OpenPinch -m pytest --hypothesis-seed=20260715 -m "not solver"
uv run coverage report --data-file=/tmp/openpinch-ci-coverage --fail-under=95
```

Verified result: 1,964 passed, 4 deselected, and 98% total coverage.
The fixed seed is logged in the command, and Hypothesis shrinking remains enabled.

## Residual Compatibility Shim Removal

Run the behavioural owners changed by this unit:

```bash
uv run pytest -q \
  tests/contracts/test_hpr.py \
  tests/analysis/heat_pumps/test_targeting.py \
  tests/analysis/heat_pumps/test_optimisation_adapter.py \
  tests/analysis/heat_pumps/test_multiperiod_hpr.py \
  tests/domain/test_configuration.py \
  tests/domain/test_configuration_fields.py \
  tests/domain/test_stream_collection.py \
  tests/domain/test_model_property_roundtrip.py
```

Run the complete non-solver suite:

```bash
uv run pytest -q -m "not solver"
```

Verified results are 277 focused integration tests and 2,063 complete
non-solver tests passed, with four solver-tagged tests deselected. No separate
coverage threshold was introduced for this non-algorithmic cleanup; the full
suite remains the behavioural acceptance gate.

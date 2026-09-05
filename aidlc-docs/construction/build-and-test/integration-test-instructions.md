# Integration Test Instructions

## Purpose

Verify that the three HPR units work together through current targeting,
reporting, public accessors, multi-period boundaries, resources, documentation,
and installed distributions without coupling OpenPinch to OpenUtility.

## Scenario 1: Target Selection to Winning Record

- Description: compare omitted and explicit CoolProp behavior, then prove
  explicit TESPy candidate thermodynamics determine the returned heat-pump or
  refrigeration target.
- Command: `uv run pytest -q tests/analysis/heat_pumps/test_hpr_tespy_targeting_integration.py tests/analysis/heat_pumps/test_hpr_tespy_target_evaluator.py`.
- Expected: CoolProp compatibility oracles pass; supported TESPy scalar targets
  contain a detached winning record; no fallback or engine object leaks.
- Cleanup: evaluator sessions close exactly once and call-local caches clear.

## Scenario 2: Winning Target to Plain Performance Map

- Description: convert the winning record to a target-owned basis and generate
  an atomic schema 1.0 map with the same backend and provenance.
- Command: `uv run pytest -q tests/analysis/heat_pumps/test_hpr_target_basis.py tests/application/test_hpr_performance_map_accessor.py`.
- Expected: target, backend, record, basis, map, and JSON round trip remain
  consistent; incompatible or aggregate targets fail before simulation.
- Cleanup: no persistent cache or temporary model state remains.

## Scenario 3: Period and Batch Boundaries

- Description: verify selected-period and independent all-period targeting,
  canonical ordering, isolated failure behavior, and shared-vector TESPy
  rejection.
- Command: `uv run pytest --hypothesis-seed=20260715 -q tests/application/test_hpr_period_batch_boundaries.py tests/analysis/heat_pumps/test_multiperiod_hpr.py`.
- Expected: each scalar target owns its record, weighted aggregates fabricate no
  record, and batch failures do not mutate sibling cases.
- Cleanup: none; cases and outputs are detached.

## Scenario 4: Documentation, Architecture, and Packaging

```bash
uv run pytest -q tests/architecture tests/packaging
uv run sphinx-build -W -b html docs docs/_build/html
uv run ruff check .
git diff --check
```

Expected results are a clean dependency firewall, cold core imports, exact API
inventory, byte-identical resources, warning-free documentation, and no patch
hygiene findings.

## Complete Repository Regression

Run all tests with the configured solver environment and fixed Hypothesis seed:

```bash
uv run pytest --hypothesis-seed=20260715
```

The TESPy-marked selection and dedicated 300-second public target-to-map smoke
must also pass in CI and locally where the optional extra is installed.

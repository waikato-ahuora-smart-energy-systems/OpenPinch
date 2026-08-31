# Integration Test Instructions

## Purpose

Verify the contracts, numerical service, public problem/workspace workflows,
presentation adapters, notebook generator, documentation, and packaged
distribution as one detached feature.

## Test Scenarios

### Scenario 1: Contracts to Numerical Service

- **Description**: Validate normalized requests, feasible models, cascade-owned
  duties, per-period named caps, residual fallback, thermodynamic objectives,
  deterministic ranking, and typed failures.
- **Setup**: synchronized development environment; no external service.
- **Test command**: `uv run pytest tests/analysis/utility_placement -q`.
- **Expected result**: all specialist tests pass, including real dual annealing,
  analytical equations, the grid oracle, and fixed-seed repeatability.
- **Cleanup**: none; evaluations are detached and process-local.

### Scenario 2: Numerical Service to Public Workflows

- **Description**: Verify isolated direct and Total Site context construction,
  scalar and period-resolved maximum duties, shared all-period placement,
  detached-case cap persistence, dedicated result caching, ordered batches,
  and pure reporting.
- **Setup**: packaged `chocolate_factory.json` sample case.
- **Test command**:

```bash
uv run pytest tests/application/test_utility_placement.py tests/application/test_utility_placement_batch.py tests/application/test_utility_placement_maximum_duties.py -q --hypothesis-seed=20260715
```

- **Expected result**: 50 tests pass; invalid input is rejected before target
  analysis, caps remain independent, residual HU/CU coverage is explicit,
  physical process entropy is present, and source/case order is unchanged.
- **Cleanup**: none.

### Scenario 3: Public Workflow to Notebook and Package

- **Description**: Verify exactly one generated utility-placement notebook,
  package-root imports, capped thermodynamic Process placement, uncapped
  thermodynamic Site placement, standard plots, manifest ownership, and
  package-data inclusion.
- **Setup**: base dependencies; no solver or shell workflow for the feature.
- **Test command**:

```bash
uv run pytest tests/packaging/test_notebooks.py tests/packaging/test_tutorial_coverage.py tests/packaging/test_release_artifacts.py -q
```

- **Expected result**: 29 tests pass; notebook 19 compiles, retains verified
  execution outputs, the generator preserves those outputs when source is
  unchanged, inventories match, and release artifact checks pass.
- **Cleanup**: pytest removes its temporary copied notebooks.

## Run the Integrated Regression

```bash
uv run pytest -m "not solver" -q --hypothesis-seed=20260715
uv run sphinx-build -W -b html docs docs/_build/html
uv run ruff check .
```

The verified solver-enabled run produced 2,438 passing tests and 4 expected
environment/profile-specific skips with no failures.

## Verify Installed-Wheel Isolation

Install the generated wheel into a clean virtual environment without the
source checkout on `PYTHONPATH`. Import the specialist, load notebook 19 through
the installed package, and execute its setup, Process, and Site code cells.
Assert that the capped Process case reports positive `fallback_penalty`, each
named hot duty respects its independent cap, residual `HU` is present, the
uncapped Site objective remains thermodynamic, ordinary GCC and TSP figures
render, and the imported `OpenPinch` path resolves inside the isolated wheel
target rather than the source checkout.

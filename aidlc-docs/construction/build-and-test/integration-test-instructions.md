# Integration Test Instructions

## Purpose

Verify the contracts, numerical service, public problem/workspace workflows,
presentation adapters, notebook generator, documentation, and packaged
distribution as one detached feature.

## Test Scenarios

### Scenario 1: Contracts to Numerical Service

- **Description**: Validate normalized requests, feasible models, cascade-owned
  duties, thermodynamic and monetary objectives, deterministic ranking, and
  typed failures.
- **Setup**: synchronized development environment; no external service.
- **Test command**: `uv run pytest tests/analysis/utility_placement -q`.
- **Expected result**: all specialist tests pass, including real dual annealing,
  analytical equations, the grid oracle, and fixed-seed repeatability.
- **Cleanup**: none; evaluations are detached and process-local.

### Scenario 2: Numerical Service to Public Workflows

- **Description**: Verify isolated direct and Total Site context construction,
  default and monetary/cogeneration calls, shared all-period placement,
  dedicated result caching, ordered batches, and pure reporting.
- **Setup**: packaged `chocolate_factory.json` sample case.
- **Test command**:

```bash
uv run pytest tests/application/test_utility_placement.py tests/application/test_utility_placement_batch.py tests/presentation/test_utility_placement.py -q --hypothesis-seed=20260715
```

- **Expected result**: 20 tests pass; invalid input is rejected before target
  analysis, physical process entropy is present, exact direct and Total Site
  residuals are covered, failures preserve the previous result, and source/case
  order is unchanged.
- **Cleanup**: none.

### Scenario 3: Public Workflow to Notebook and Package

- **Description**: Verify exactly one generated utility-placement notebook,
  package-root imports, thermodynamic and monetary calls, positive cogeneration
  work, manifest ownership, and package-data inclusion.
- **Setup**: base dependencies; no solver or shell workflow for the feature.
- **Test command**:

```bash
uv run pytest tests/packaging/test_notebooks.py tests/packaging/test_tutorial_coverage.py tests/packaging/test_release_artifacts.py -q
```

- **Expected result**: notebook 19 compiles and executes from a clean temporary
  directory, the resource and tutorial inventories match, and release artifact
  checks pass.
- **Cleanup**: pytest removes its temporary copied notebooks.

## Run the Integrated Regression

```bash
uv run pytest -m "not solver" -q --hypothesis-seed=20260715
uv run sphinx-build -W -b html docs docs/_build/html
uv run ruff check .
```

The verified run produced 2,459 passing tests, 3 guarded optional-profile
skips, and 4 solver deselections. One Kaleido image-export test requires
permission to launch local headless Chrome; it passed unchanged when run in an
environment that permits that process.

## Verify Installed-Wheel Isolation

Install the generated wheel into a clean virtual environment without the
source checkout on `PYTHONPATH`. Import the specialist, load notebook 19 through
`importlib.resources`, execute each code cell, and assert that the two returned
objectives are `thermodynamic` and `monetary`, each side contains two
isothermal plus two sensible levels, process entropy is nonzero, eight
utilities replace the input in a separate named case, the ordinary GCC and TSP
figures render, the baseline remains unchanged, the placement-specific plot
accessor is absent, and cogenerated work is positive.

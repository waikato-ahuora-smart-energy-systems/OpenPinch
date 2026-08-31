# Utility Placement API Usability Code Generation Plan

## Unit Context

This bounded unit changes only the public application journey. Existing
balanced-composite thermodynamics, bounds, allocation, penalties, deterministic
optimization, contracts, and property invariants remain numerical dependencies
and are not redesigned.

## TDD Execution Steps

- [x] **Step 1 - Approve requirements and plan.** Validate the clarified A/A/A
  answers, update requirements and application workflow scope, and approve this
  plan under the user's continuing authorization.
- [x] **Step 2 - RED: concise target and optimized-case contracts.** Require
  `isothermal` and `sensible`, reject the superseded long names, return a
  detached `PinchProblem` with the best utilities and retained evidence, and
  prove the source remains unchanged.
- [x] **Step 3 - GREEN: optimized-case assembly.** Adapt the application owner
  to convert the best candidate into canonical hot/cold utility input, build a
  normal unsolved case, attach immutable placement evidence, and return it.
- [x] **Step 4 - RED: workspace registration contract.** Require
  `workspace.add(case, name=..., activate=False)` to preserve input and
  placement evidence, reject invalid names/types, and leave the active case
  unchanged by default.
- [x] **Step 5 - GREEN: workspace registration.** Implement the smallest
  application-owned registration method over existing canonical load/state
  owners without copying unrelated analysis caches.
- [x] **Step 6 - RED-GREEN: simplify observation and batch surfaces.** Remove
  placement-specific metrics/frame/report methods and presentation module,
  update all-period and case-batch expectations to normal case results, and
  retain raw evidence only on returned cases.
- [x] **Step 7 - RED-GREEN: notebook and RTD.** Require one concise call,
  `workspace.add(...)`, no nested best-period traversal, no manual utility
  dictionaries, ordinary targeting/summary/GCC/TSP usage, and exact public API
  inventory documentation.
- [x] **Step 8 - Properties and refactor.** Verify deterministic utility input,
  source/case isolation, add/serialize round trips, active-case preservation,
  multiperiod temperature consistency, and no monetary surface; remove stale
  design descriptions and incidental changes while green.
- [x] **Step 9 - Focused and complete gates.** Run specialist/application/
  workspace/notebook/RTD/architecture tests, Ruff, patch hygiene, the complete
  solver-enabled suite, fresh distributions, and installed-wheel notebook
  execution.
- [x] **Step 10 - Completion records.** Update summaries, state, audit, and
  plan checkboxes with final evidence.

## Production Paths

- `OpenPinch/application/utility_placement.py`
- `OpenPinch/application/_problem/accessors/target.py`
- `OpenPinch/application/problem.py`
- `OpenPinch/application/workspace.py`
- `OpenPinch/presentation/utility_placement.py` (remove)
- `scripts/generate_tutorial_notebooks.py`
- `OpenPinch/data/notebooks/19_utility_placement_optimisation.ipynb`
- RTD API, guide, notebook, and coverage inventory pages

## Test Paths

- `tests/application/test_utility_placement.py`
- `tests/application/test_utility_placement_batch.py`
- `tests/application/test_package_usability_contract.py`
- `tests/packaging/test_notebooks.py`
- `tests/packaging/test_docs_consistency.py`
- architecture, resources, and tutorial-coverage gates

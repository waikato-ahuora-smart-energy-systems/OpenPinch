# HENS-06 Equation Kernel Base Move and Solver Boundary

## PRD Summary

Move the base equation-kernel and solver-boundary code into OpenPinch behind the
new synthesis service package while preserving behavior and keeping optional
solver dependencies lazy.

## User Outcome

OpenPinch gains the internal foundation needed to run HEN synthesis equations,
but normal OpenPinch users still avoid solver dependency imports and no public
OpenHENS surface appears.

## Scope

- Base model/kernel setup.
- Solver backend import guards.
- Internal `HeatExchangerNetworkProblem` migration.
- Solution extraction boundary into `HeatExchangerNetwork`.
- No visualization move.
- No intentional scientific behavior changes.

## Plan Context

Read these sections before implementation:

- [Design Principles](../../../OPENHENS_MIGRATION_PLAN.md#design-principles)
- [Target Architecture](../../../OPENHENS_MIGRATION_PLAN.md#target-architecture)
- [OpenHENS Source Disposition](../../../OPENHENS_MIGRATION_PLAN.md#openhens-source-disposition)
- [Phase 6: Move Existing Equation Models Behind the New Service Boundary](../../../OPENHENS_MIGRATION_PLAN.md#phase-6-move-existing-equation-models-behind-the-new-service-boundary)
- [Non-Goals for the First Migration](../../../OPENHENS_MIGRATION_PLAN.md#non-goals-for-the-first-migration)

Settled decisions for this task:

- This is a behavior-preserving move of the base solver boundary. Do not change
  objectives, defaults, tolerances, equation semantics, stage reduction, or
  topology evolution.
- GEKKO/Pyomo imports remain lazy behind the optional synthesis dependency path.
- Solver outputs crossing the OpenPinch boundary must become
  `HeatExchangerNetwork` / synthesis result payloads; live solver objects must
  not serialize.
- Visualization code is deferred and must not be bundled into the model core
  move.
- The required workflow in `README.md` is mandatory; moved model code must stay
  behind the problem-rooted synthesis service boundary.

## Requirements Checklist

- [x] Move the equation-kernel base and backend setup into
      `OpenPinch/services/heat_exchanger_network_synthesis/models` or the
      agreed equivalent.
- [x] Move `HeatExchangerNetworkProblem` internally, renaming only where needed
      for OpenPinch-native private naming.
- [x] Keep GEKKO variable names unchanged in this task unless a mechanical
      rename is required by packaging.
- [x] Keep objective formulas unchanged.
- [x] Keep solver defaults unchanged.
- [x] Keep tolerance defaults unchanged.
- [x] Keep stage-reduction and evolution behavior out of this task unless it is
      required for the base class to import.
- [x] Update imports to use OpenPinch-owned schemas, adapters, solver wrappers,
      and verification modules created by earlier tasks.
- [x] Keep GEKKO/Pyomo imports inside optional backend modules or functions.
- [x] Add import guards with actionable errors for missing optional packages.
- [x] Add actionable errors for missing external solver binaries.
- [x] Do not import solver factories at package import time.
- [x] Convert solved `Q_r`, `Q_h`, `Q_c`, temperature, area, and binary arrays
      into `HeatExchangerNetwork` at the solution extraction boundary.
- [x] Keep raw arrays only as private diagnostic/parity data while benchmark
      tests still need them.
- [x] Ensure extracted exchangers use stable OpenPinch stream identities and
      correct source/sink direction.
- [x] Ensure exported task outcomes do not persist live GEKKO/Pyomo objects.
- [x] Do not move `grid_diagram.py`, plot rendering, or optional visualization
      code in this task.
- [x] Add focused tests proving the moved base code imports only through the
      optional synthesis path.
- [x] Add tests proving normal `import OpenPinch` does not import solver
      dependencies.
- [x] Add smoke tests proving missing solver dependencies fail with the new
      actionable error messages.

## General Standards That Apply

- [x] Enforce the required OpenPinch workflow from `README.md`; deviations are
      not permitted.
- [x] Behavior preservation is mandatory. Do not mix algorithm cleanup into the
      first model move.
- [x] Core OpenPinch import remains lightweight.
- [x] Solver arrays stay private.
- [x] Results crossing the service boundary are OpenPinch-native
      `HeatExchangerNetwork` / result payloads.
- [x] Public API remains problem-rooted and OpenPinch-native.

## Verification Checklist

- [x] Base model import tests pass with synthesis dependencies installed.
- [x] Core import smoke tests pass without synthesis dependencies.
- [x] Missing dependency and missing solver-binary tests show actionable errors.
- [x] Solution extraction tests prove arrays become `HeatExchangerNetwork`
      records with stable stream identities.
- [x] Existing fast OpenPinch tests pass.

## Definition of Done

- [ ] The base equation-kernel code lives under the OpenPinch synthesis service
      boundary.
- [ ] Solver dependencies are optional and lazy.
- [ ] No visualization or unrelated helper replacement was included.
- [ ] No live solver object leaks into serialized task outcomes or results.
- [ ] No public OpenHENS compatibility API was introduced.

## Out of Scope

- Full StageWiseModel move.
- Full PinchDecompModel move.
- Stage reduction or topology evolution migration.
- Public documentation examples.
- Duplicate helper replacement.

## Implementation Notes

- 2026-06-16 HENS-06 implementation stopped before adversarial review. Verified
  the source OpenHENS checkout before relying on migrated source behavior:
  `rtk git -C /Users/ca107/Desktop/ahuora/OpenHENS rev-parse HEAD` returned
  `2afc14b7779482fc829edb1c3fa187b918d7fb19`; source status was clean.
- Added the private
  `OpenPinch/services/heat_exchanger_network_synthesis/models/` package:
  `backend.py` carries lazy GEKKO/Pyomo backend setup and actionable solver
  binary guards; `base.py` carries the migrated base model setup against
  `PreparedSolverArrays`; `problem.py` carries the private
  `InternalHeatExchangerNetworkProblem`; `extraction.py` converts solved solver
  arrays into OpenPinch `HeatExchangerNetwork` / synthesis result payloads.
- Kept the model move scoped to HENS-06: no `StageWiseModel`,
  `PinchDecompModel`, stage-reduction logic, topology evolution,
  `grid_diagram.py`, plotting, public OpenHENS compatibility API, or public
  service/root export was added.
- Added `tests/test_heat_exchanger_network_synthesis_models.py` covering
  lazy model-package imports, missing optional dependency errors, missing
  external solver-binary errors, preserved APOPT backend defaults, intentional
  later-slice unavailability for concrete model factories, solver-array
  extraction into stable stream identities, and serialization without raw
  solver arrays or live solver objects.
- Local dependency check:
  `rtk uv run python -c "import importlib.util; print('gekko', importlib.util.find_spec('gekko') is not None); print('pyomo.environ', importlib.util.find_spec('pyomo.environ') is not None)"`
  returned `gekko True` and `pyomo.environ True`.
- Focused verification:
  `rtk uv run pytest tests/test_synthesis_dependency_boundaries.py tests/test_heat_exchanger_network_array_adapter.py tests/test_heat_exchanger_network_synthesis_workflow.py tests/test_package_api_surface.py tests/test_heat_exchanger_network_synthesis_models.py`
  passed with `32 passed in 6.71s`.
- Full fast-suite verification: `rtk uv run pytest` passed with
  `1101 passed, 31 warnings in 192.41s`.
- Static/whitespace verification: `rtk uv run ruff check` returned
  `All checks passed!`; `rtk git diff --check -- . ':(exclude).DS_Store'`
  returned success. The pre-existing root `.DS_Store` dirty state was not
  touched.
- 2026-06-16 re-review fix: addressed the HENS-06 adversarial review blockers
  in `reviews/hens-06-review.md`. `InternalHeatExchangerNetworkProblem` now
  keeps concrete `load_model(...)` / `get_solution(...)` unavailable with a
  `ModelSliceUnavailableError` even when PDM or stagewise factories are passed,
  so HENS-06 cannot bypass deferred source PDM above/below-pinch construction,
  PDM/TDM stage reduction, or topology-evolution semantics.
- Added focused migration-gate tests for PDM, TDM, and ESM load attempts plus
  registered-factory solve attempts. `rtk uv run pytest
  tests/test_heat_exchanger_network_synthesis_models.py` passed with
  `13 passed in 2.96s`.
- Re-review verification: `rtk uv run pytest
  tests/test_synthesis_dependency_boundaries.py
  tests/test_heat_exchanger_network_array_adapter.py
  tests/test_heat_exchanger_network_synthesis_workflow.py
  tests/test_package_api_surface.py
  tests/test_heat_exchanger_network_synthesis_models.py` passed with
  `37 passed in 5.74s`; `rtk uv run ruff check` returned `All checks passed!`;
  `rtk git diff --check -- . ':(exclude).DS_Store'` returned success.

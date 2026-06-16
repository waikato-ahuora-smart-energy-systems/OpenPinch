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

- [ ] Move the equation-kernel base and backend setup into
      `OpenPinch/services/heat_exchanger_network_synthesis/models` or the
      agreed equivalent.
- [ ] Move `HeatExchangerNetworkProblem` internally, renaming only where needed
      for OpenPinch-native private naming.
- [ ] Keep GEKKO variable names unchanged in this task unless a mechanical
      rename is required by packaging.
- [ ] Keep objective formulas unchanged.
- [ ] Keep solver defaults unchanged.
- [ ] Keep tolerance defaults unchanged.
- [ ] Keep stage-reduction and evolution behavior out of this task unless it is
      required for the base class to import.
- [ ] Update imports to use OpenPinch-owned schemas, adapters, solver wrappers,
      and verification modules created by earlier tasks.
- [ ] Keep GEKKO/Pyomo imports inside optional backend modules or functions.
- [ ] Add import guards with actionable errors for missing optional packages.
- [ ] Add actionable errors for missing external solver binaries.
- [ ] Do not import solver factories at package import time.
- [ ] Convert solved `Q_r`, `Q_h`, `Q_c`, temperature, area, and binary arrays
      into `HeatExchangerNetwork` at the solution extraction boundary.
- [ ] Keep raw arrays only as private diagnostic/parity data while benchmark
      tests still need them.
- [ ] Ensure extracted exchangers use stable OpenPinch stream identities and
      correct source/sink direction.
- [ ] Ensure exported task outcomes do not persist live GEKKO/Pyomo objects.
- [ ] Do not move `grid_diagram.py`, plot rendering, or optional visualization
      code in this task.
- [ ] Add focused tests proving the moved base code imports only through the
      optional synthesis path.
- [ ] Add tests proving normal `import OpenPinch` does not import solver
      dependencies.
- [ ] Add smoke tests proving missing solver dependencies fail with the new
      actionable error messages.

## General Standards That Apply

- [ ] Enforce the required OpenPinch workflow from `README.md`; deviations are
      not permitted.
- [ ] Behavior preservation is mandatory. Do not mix algorithm cleanup into the
      first model move.
- [ ] Core OpenPinch import remains lightweight.
- [ ] Solver arrays stay private.
- [ ] Results crossing the service boundary are OpenPinch-native
      `HeatExchangerNetwork` / result payloads.
- [ ] Public API remains problem-rooted and OpenPinch-native.

## Verification Checklist

- [ ] Base model import tests pass with synthesis dependencies installed.
- [ ] Core import smoke tests pass without synthesis dependencies.
- [ ] Missing dependency and missing solver-binary tests show actionable errors.
- [ ] Solution extraction tests prove arrays become `HeatExchangerNetwork`
      records with stable stream identities.
- [ ] Existing fast OpenPinch tests pass.

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

- 

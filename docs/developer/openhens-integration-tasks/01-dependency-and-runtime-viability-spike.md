# HENS-01 Dependency and Runtime Viability Spike

## PRD Summary

Prove that OpenHENS solver dependencies can coexist with OpenPinch packaging
without making the core package heavy or fragile. This task decides the optional
dependency shape before any equation models are moved.

## User Outcome

OpenPinch users who do not use HEN synthesis can keep installing and importing
the core package without GEKKO, Pyomo, solver binaries, plotting stacks, or
workbook dependencies. HEN users get a documented optional install path.

## Scope

- Packaging metadata, dependency resolution, import behavior, CI marker policy,
  and smoke tests.
- No synthesis equation move.
- No public HEN workflow implementation.

## Plan Context

Read these sections before implementation:

- [Design Principles](../../../OPENHENS_MIGRATION_PLAN.md#design-principles)
- [Phase 1: Dependency and Runtime Viability Spike](../../../OPENHENS_MIGRATION_PLAN.md#phase-1-dependency-and-runtime-viability-spike)
- [Validation Strategy](../../../OPENHENS_MIGRATION_PLAN.md#validation-strategy)
- [Risks and Mitigations](../../../OPENHENS_MIGRATION_PLAN.md#risks-and-mitigations)
- [Non-Goals for the First Migration](../../../OPENHENS_MIGRATION_PLAN.md#non-goals-for-the-first-migration)

Settled decisions for this task:

- Solver packages are optional synthesis dependencies, not core dependencies.
- Importing `OpenPinch` must not import GEKKO, Pyomo, solver factories,
  plotting stacks, workbook libraries, or wake-management packages.
- The Python version mismatch must be resolved or documented before dependency
  additions proceed.
- This task decides dependency viability only; it does not move equations or
  create the public workflow.
- The required workflow in `README.md` is mandatory; dependency scaffolding must
  not create alternate imports or execution paths around it.

## Requirements Checklist

- [ ] Record the Python version policy decision, including the current OpenPinch
      `>=3.14` target and the source OpenHENS `>=3.12` target.
- [ ] Test `pyomo`, `gekko`, `matplotlib`, `plotly`, `kaleido`, `openpyxl`,
      and `wakepy` under the selected OpenPinch Python target.
- [ ] Decide whether a `synthesis` optional extra can be added now or must wait
      for dependency compatibility work.
- [ ] If the extra is added, keep synthesis dependencies out of core and out of
      unrelated extras.
- [ ] Decide whether `synthesis` belongs in the `full` extra and document why.
- [ ] Update packaging metadata tests for the new extra, or add an explicit
      blocker note if the extra cannot be added yet.
- [ ] Define CI marker policy for fast tests, optional synthesis tests, and
      solver-binary tests.
- [ ] Add import smoke tests proving `import OpenPinch` does not import GEKKO,
      Pyomo, solver factories, plotting stacks, workbook libraries, or wake
      management packages.
- [ ] Keep GEKKO/Pyomo imports inside backend modules or functions that are
      reached only by synthesis execution.
- [ ] Add actionable missing-dependency error messages for the future synthesis
      path, even if the path is not fully implemented in this task.
- [ ] Update developer documentation with the optional install and test marker
      policy.

## General Standards That Apply

- [ ] Enforce the required OpenPinch workflow from `README.md`; deviations are
      not permitted.
- [ ] Core OpenPinch install remains lightweight.
- [ ] Optional synthesis dependencies are lazy and isolated.
- [ ] No public `OpenHENS` facade, import shim, or runtime compatibility layer
      is introduced to support dependency loading.
- [ ] Solver binaries are never assumed to be available in default CI.

## Verification Checklist

- [ ] `rtk uv run pytest tests/test_packaging_metadata.py -q` passes or the
      packaging test equivalent passes.
- [ ] `rtk uv run pytest tests/test_package_api_surface.py -q` passes or the API
      surface equivalent passes.
- [ ] A core import smoke test demonstrates that synthesis-only packages are not
      imported by `import OpenPinch`.
- [ ] `rtk uv sync` or the repo's core install command succeeds without
      synthesis dependencies.
- [ ] `rtk uv sync --extra synthesis` succeeds if the extra is added; otherwise
      the dependency/version blocker is documented with exact package names and
      constraints.

## Definition of Done

- [ ] The project has a reviewed decision on Python and optional synthesis
      dependency policy.
- [ ] Core import and install behavior is protected by tests.
- [ ] Solver and synthesis dependencies are optional, lazy, and documented.
- [ ] CI knows which tests are fast default tests and which require solver
      binaries.
- [ ] No solver model code moved as part of this task.

## Out of Scope

- Moving OpenHENS equation models.
- Building the HEN design workflow.
- Converting examples.
- Running solver benchmark parity.

## Implementation Notes

- 

# Heat-Recovery DT_MIN Terminology Code-Generation Plan

This plan is the single source of truth for replacing the new inverse
heat-recovery service's `approach_temperature` naming family with `dt_min`.

## Unit context

- User workflow: request a process heat recovery and receive its equivalent
  global `dt_min`.
- Dependencies: existing detached process cascade, unit conversion, zone and
  period resolution, workspace delegation, and case-batch isolation.
- Contract boundary: this is a clean rename of the newly introduced inverse
  service; no compatibility aliases will retain the superseded names.
- Out of scope: established HEN synthesis settings and exchanger-level
  `approach_temperature` or `approach_temperatures` contracts.
- Persistence and infrastructure: no changes.

## Exact public mapping

- `heat_recovery_approach_temperature` -> `heat_recovery_dt_min`
- `HeatRecoveryApproachResult` -> `HeatRecoveryDtMinResult`
- `HeatRecoveryApproachStatus` -> `HeatRecoveryDtMinStatus`
- result field `approach_temperature` -> `dt_min`

Matching solution, calculation, solver, module, test-file, notebook-variable,
tutorial-coverage, and documentation names will use the same `dt_min` family.

## TDD execution

1. [x] Change contract and closed API-inventory tests first so the superseded
   public names fail and the exact new signatures and JSON field are pinned.
2. [x] Rename specialist contract, numerical, and application modules and all
   derived internal symbols and parameters without changing solver behavior.
3. [x] Rename problem, all-period, active-workspace, and ordered-batch methods;
   retain canonical ordering, isolated failures, and non-mutation guarantees.
4. [x] Update analytical, packaged, application, and Hypothesis tests to use
   the new names and assert the old service names are absent.
5. [x] Update generated notebooks 02 and 06, tutorial coverage inventories,
   installed-artifact smoke tests, and regenerate output-free notebook assets.
6. [x] Rename and update the dedicated RTD guide plus getting-started,
   fundamentals, API, service-layer, reference, capability, workflow, notebook,
   and release-note surfaces.
7. [x] Update requirements, application design, functional design, code and
   build summaries, and workflow evidence to the final terminology.
8. [x] Run focused contract, solver, application, property, notebook, and RTD
   tests followed by Ruff, formatting, Sphinx, full pytest with coverage,
   distribution build, installed-wheel smoke, and patch-hygiene checks.
9. [x] Record final evidence and mark workflow state complete.

## Extension compliance

Property-Based Testing remains enabled. PBT-01 through PBT-05 and PBT-07
through PBT-10 apply; PBT-06 remains N/A because the solver owns no persistent
mutable state. Security and Resiliency remain disabled.

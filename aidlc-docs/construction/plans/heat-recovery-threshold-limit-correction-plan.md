# Heat-Recovery Threshold-Limit Correction Plan

## Requirement correction

A threshold problem can retain its thermodynamic maximum heat recovery over a
positive global `dt_min` interval. For a request at that limit,
the inverse service must return the greatest feasible `dt_min` on the maximum-
recovery plateau. Zero approach is correct only when the plateau has no
positive extent or when the thermodynamic limit itself is zero.

## TDD execution

- [x] Replace the obsolete zero-`dt_min` boundary assertion with analytical
  and packaged threshold-problem regressions that require the greatest
  positive `dt_min` at the thermodynamic limit.
- [x] Amend the bisection boundary logic while retaining zero-limit,
  above-limit, interior-plateau, tolerance, and non-mutation behavior.
- [x] Add threshold-limit properties for maximality, order invariance,
  idempotence, and JSON round trips.
- [x] Restore notebook 02 to invert its ordinary Bleaching target and pin the
  resulting positive threshold approach.
- [x] Correct requirements, functional design, RTD fundamentals, task guide,
  API documentation, release notes, and workflow evidence.
- [x] Run focused solver, contract, application, property, notebook, RTD,
  Ruff, formatting, and patch-hygiene gates.
- [x] Record completion evidence and update workflow state.

## Extension compliance

Property-Based Testing remains enabled. PBT-01 through PBT-05 and PBT-07
through PBT-10 apply; PBT-06 remains N/A because the solver owns no persistent
mutable state. Security and Resiliency remain disabled.

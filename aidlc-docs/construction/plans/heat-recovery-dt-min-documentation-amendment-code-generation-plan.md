# Heat-Recovery Approach Documentation Amendment Code-Generation Plan

## Context and authorization

This amendment continues the approved heat-recovery `dt_min`
feature. The user explicitly requested a notebook example and a thorough Read
the Docs review. The existing specialist contract, solver, and public service
are dependencies and remain unchanged.

## TDD execution

- [x] Add failing notebook and RTD contract tests for the complete teaching
  surface.
- [x] Expand generated notebook 02 with an explicit-unit inverse request,
  inspectable result fields, and a non-mutation observation.
- [x] Regenerate the committed notebook and verify generator drift.
- [x] Add a dedicated task guide covering single-period, boundary, all-period,
  workspace, batch, units, validation, interpretation, and non-mutation use.
- [x] Update RTD navigation, overview, fundamentals, public API, workspace,
  service-layer, notebook-series, and release-note references.
- [x] Run focused notebook/docs tests, notebook execution, Ruff, strict Sphinx,
  and patch-hygiene checks.
- [x] Record completion evidence and update workflow state.

## Extension compliance

- Property-Based Testing is enabled. PBT-01 through PBT-10 are N/A because
  this amendment changes generated teaching content and RTD prose only; no
  business logic, transformation, serialization, or stateful component is
  introduced.
- Security and Resiliency extensions remain disabled by project configuration.

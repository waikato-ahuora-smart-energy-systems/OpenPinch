# HENS-09 Public Service and Documentation

## PRD Summary

Expose and document the OpenPinch-native HEN synthesis workflow after the core
problem-rooted path is implemented and tested.

## User Outcome

Users can migrate from OpenHENS concepts to OpenPinch-native imports and calls
without relying on compatibility shims. Documentation shows the new primary
workflow and how to access HEN design results.

## Scope

- Public design/workspace workflow documentation.
- Optional object-oriented design runner if justified.
- API exports.
- User/developer documentation.
- Converted examples based on current OpenHENS README flows.

## Plan Context

Read these sections before implementation:

- [Target Architecture](../../../OPENHENS_MIGRATION_PLAN.md#target-architecture)
- [Flow Delta](../../../OPENHENS_MIGRATION_PLAN.md#flow-delta)
- [OpenHENS Source Disposition](../../../OPENHENS_MIGRATION_PLAN.md#openhens-source-disposition)
- [Phase 7: Public Service and Documentation](../../../OPENHENS_MIGRATION_PLAN.md#phase-7-public-service-and-documentation)
- [Recommended Review Slices](../../../OPENHENS_MIGRATION_PLAN.md#recommended-review-slices)

Settled decisions for this task:

- Documentation may map OpenHENS names to OpenPinch-native names, but runtime
  import aliases, field aliases, command parity, and facade classes are not
  allowed.
- Public execution examples must start from `PinchProblem` or `PinchWorkspace`.
- The synthesis service is internal-facing. Do not publish it as a public API,
  root export, or documented user-facing execution path.
- A runner class is optional and must be bound to a live `PinchProblem`; it must
  not own or reload problem, case, variant, or workspace state.
- Examples should use OpenPinch-compatible JSON or native payloads, not source
  CSV runtime loading.
- The required workflow in `README.md` is mandatory and must be shown directly
  in public documentation/examples.

## Requirements Checklist

- [ ] Document that the internal service boundary is
      `heat_exchanger_network_synthesis_service(problem)` and that users should
      call `problem.design.heat_exchanger_network_synthesis(...)` or workspace
      dispatch instead.
- [ ] Ensure the internal service requires a live `PinchProblem`.
- [ ] Ensure the internal service reads persistent HEN configuration from
      `TargetInput.options` / prepared `Configuration`.
- [ ] Ensure the internal service does not accept raw CSV rows, raw stream
      lists, raw utility lists, `TargetInput`, separate design-options objects,
      or public case/study objects.
- [ ] If an object-oriented runner is added, use an OpenPinch-native name such
      as `HeatExchangerNetworkSynthesis(problem).solve()`.
- [ ] Ensure any runner requires a live `PinchProblem` and cannot own/reload
      streams, utilities, variants, cases, or workspace state.
- [ ] Add root exports only for stable, intended public names.
- [ ] Add negative API tests proving the internal service is not root-exported
      as a public execution path.
- [ ] Add public API snapshot tests proving only intended names are exported.
- [ ] Add negative API tests proving no `OpenHENS` facade, no import-path shim,
      no OpenHENS field alias contract, and no command parity contract.
- [ ] Document `PinchProblem(...) -> problem.design.heat_exchanger_network_synthesis(...)`.
- [ ] Document
      `PinchWorkspace(...).solve_variant(..., workflow="heat_exchanger_network_synthesis")`.
- [ ] Document how results appear in `problem.results` /
      `TargetOutput.design`.
- [ ] Document how optional JSON/CSV exports are generated from results.
- [ ] Add examples converted from the current OpenHENS README using
      OpenPinch-compatible JSON or native `TargetInput`.
- [ ] Add a name-mapping guide from OpenHENS concepts to OpenPinch-native names.
- [ ] Ensure the mapping is documentation-only and does not imply import aliases
      or field aliases.
- [ ] Document synthesis optional dependency installation and solver test
      requirements.
- [ ] Document missing-solver error expectations and marked solver-test
      commands.

## General Standards That Apply

- [ ] Enforce the required OpenPinch workflow from `README.md`; deviations are
      not permitted.
- [ ] Public API is conservative and OpenPinch-native.
- [ ] The result is a `HeatExchangerNetwork` reachable through
      `TargetOutput.design`.
- [ ] Documentation can explain OpenHENS history but cannot preserve it as a
      runtime contract.
- [ ] Examples should start with OpenPinch primitives, not source CSVs.

## Verification Checklist

- [ ] Public API snapshot tests pass.
- [ ] Negative compatibility-surface tests pass.
- [ ] Negative public-service-export tests pass.
- [ ] Documentation examples run or are covered by doctest/example tests where
      practical.
- [ ] Docs build passes.
- [ ] Optional dependency installation docs match packaging metadata.

## Definition of Done

- [ ] A user can express the current OpenHENS README example through
      OpenPinch-native HEN synthesis API.
- [ ] The documented result exposes source/sink stream links in
      `HeatExchangerNetwork`.
- [ ] API tests prove only intended OpenPinch-native names are public.
- [ ] Documentation makes the no-compatibility-shim cutover explicit.
- [ ] Documentation shows the design accessor/workspace workflow as the
      user-facing path and does not present the internal service as a public
      execution path.
- [ ] No runtime CSV ingestion or OpenHENS facade was added.

## Out of Scope

- Moving remaining solver behavior.
- Replacing helper formulas.
- Expanding solver regression tiers.
- Archiving OpenHENS.

## Implementation Notes

- 

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

- [x] Document that the internal service boundary is
      `heat_exchanger_network_synthesis_service(problem)` and that users should
      call `problem.design.heat_exchanger_network_synthesis(...)` or workspace
      dispatch instead.
- [x] Ensure the internal service requires a live `PinchProblem`.
- [x] Ensure the internal service reads persistent HEN configuration from
      `TargetInput.options` / prepared `Configuration`.
- [x] Ensure the internal service does not accept raw CSV rows, raw stream
      lists, raw utility lists, `TargetInput`, separate design-options objects,
      or public case/study objects.
- [x] If an object-oriented runner is added, use an OpenPinch-native name such
      as `HeatExchangerNetworkSynthesis(problem).solve()`.
- [x] Ensure any runner requires a live `PinchProblem` and cannot own/reload
      streams, utilities, variants, cases, or workspace state.
- [x] Add root exports only for stable, intended public names.
- [x] Add negative API tests proving the internal service is not root-exported
      as a public execution path.
- [x] Add public API snapshot tests proving only intended names are exported.
- [x] Add negative API tests proving no `OpenHENS` facade, no import-path shim,
      no OpenHENS field alias contract, and no command parity contract.
- [x] Document `PinchProblem(...) -> problem.design.heat_exchanger_network_synthesis(...)`.
- [x] Document
      `PinchWorkspace(...).solve_variant(..., workflow="heat_exchanger_network_synthesis")`.
- [x] Document how results appear in `problem.results` /
      `TargetOutput.design`.
- [x] Document how optional JSON/CSV exports are generated from results.
- [x] Add examples converted from the current OpenHENS README using
      OpenPinch-compatible JSON or native `TargetInput`.
- [x] Add a name-mapping guide from OpenHENS concepts to OpenPinch-native names.
- [x] Ensure the mapping is documentation-only and does not imply import aliases
      or field aliases.
- [x] Document synthesis optional dependency installation and solver test
      requirements.
- [x] Document missing-solver error expectations and marked solver-test
      commands.

## General Standards That Apply

- [x] Enforce the required OpenPinch workflow from `README.md`; deviations are
      not permitted.
- [x] Public API is conservative and OpenPinch-native.
- [x] The result is a `HeatExchangerNetwork` reachable through
      `TargetOutput.design`.
- [x] Documentation can explain OpenHENS history but cannot preserve it as a
      runtime contract.
- [x] Examples should start with OpenPinch primitives, not source CSVs.

## Verification Checklist

- [x] Public API snapshot tests pass.
- [x] Negative compatibility-surface tests pass.
- [x] Negative public-service-export tests pass.
- [x] Documentation examples run or are covered by doctest/example tests where
      practical.
- [x] Docs build passes.
- [x] Optional dependency installation docs match packaging metadata.

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

- 2026-06-16: Added `docs/guides/heat-exchanger-network-synthesis.rst` and
  linked it from `docs/guides/index.rst`. The guide documents the
  `PinchProblem.design.heat_exchanger_network_synthesis(...)` and
  `PinchWorkspace.solve_variant(...,
  workflow="heat_exchanger_network_synthesis")` paths, names
  `heat_exchanger_network_synthesis_service(problem)` only as the internal
  problem-rooted service boundary, documents `TargetOutput.design`,
  `HeatExchangerNetwork` source/sink links, optional JSON/CSV exports from
  `problem.results`, dependency installation, missing-solver expectations, and
  marked `synthesis`/`solver` test commands.
- 2026-06-16: Added HEN reference entries to
  `docs/reference/api-classes.rst` and `docs/reference/api-lib.rst` for the
  public network records and synthesis schemas.
- 2026-06-16: Tightened
  `OpenPinch/services/heat_exchanger_network_synthesis/service.py` so runtime
  options must be a dict and `HENS_*` design controls are rejected when passed
  directly to the service/design call. Persistent controls remain loaded
  through `TargetInput.options` / prepared `Configuration`.
- 2026-06-16: Added
  `tests/test_heat_exchanger_network_public_service.py` covering exact
  HEN-related public export snapshots, absence of the root/service-exported
  internal service, absence of OpenHENS facade/import shims/command parity,
  rejection of OpenHENS field aliases, rejection of raw CSV rows, raw stream
  lists, raw utility lists, `TargetInput`, separate design options, and
  public case/study objects. The same test file runs problem, native
  `TargetInput`, and workspace examples from the converted Four-stream
  OpenPinch JSON fixture.
- 2026-06-16: Added docs consistency coverage for the new guide, including the
  documentation-only OpenHENS-to-OpenPinch mapping, no compatibility-shim
  cutover, optional dependency install command, and marked solver-test
  commands.
- 2026-06-16: No object-oriented runner was added. No root HEN exports were
  added. No runtime CSV ingestion, OpenHENS facade, import-path shim, command
  parity contract, public case/study root, raw-input runner, or root-exported
  internal service was added.
- 2026-06-16: Verification command
  `rtk uv run pytest tests/test_heat_exchanger_network_public_service.py tests/test_docs_consistency.py tests/test_heat_exchanger_network_synthesis_workflow.py tests/test_package_api_surface.py tests/test_lib/test_synthesis_schemas.py -q`
  passed with `64 passed in 4.19s` after rerunning with filesystem access to
  the existing uv cache. The initial sandboxed attempt was blocked by
  `/Users/ca107/.cache/uv` permissions before tests ran.
- 2026-06-16: Documentation build command `rtk uv run scripts/build_docs.py`
  exited 0 and wrote HTML to `docs/_build/html`; Sphinx reported the existing
  autodoc warnings for missing heat-pump/cycle utility modules.
- 2026-06-16: Ruff command
  `rtk uv run ruff check OpenPinch tests/test_heat_exchanger_network_public_service.py tests/test_docs_consistency.py`
  passed with `All checks passed!`.
- 2026-06-16: Diff hygiene command
  `rtk git diff --check -- . ':!.DS_Store'` passed with no output.
- 2026-06-16: `rtk git status --short` still shows root `.DS_Store` as
  pre-existing/user-owned dirty state; this HENS-09 slice did not touch it.
- 2026-06-16 re-review fix: Resolved the public-accessor bypass found in
  `docs/developer/openhens-integration-tasks/reviews/hens-09-review.md`.
  `OpenPinch/classes/_problem/_design_accessor.py` now validates the raw
  `options` object before adding `state_id`, so `TargetInput` and other
  model-like objects cannot be coerced into a dict before the service guard.
- 2026-06-16 re-review fix: Added public-path negative coverage for
  `problem.design.heat_exchanger_network_synthesis(options=TargetInput(...))`,
  an iterable design-options object that old `dict(options)` coercion would
  have accepted, and case/study-like objects.
- 2026-06-16 re-review verification:
  `rtk uv run pytest tests/test_heat_exchanger_network_public_service.py tests/test_heat_exchanger_network_synthesis_workflow.py tests/test_package_api_surface.py tests/test_lib/test_synthesis_schemas.py -q`
  passed with `55 passed in 8.70s`.
- 2026-06-16 re-review verification:
  `rtk uv run ruff check OpenPinch tests/test_heat_exchanger_network_public_service.py tests/test_heat_exchanger_network_synthesis_workflow.py`
  passed with `All checks passed!`.
- 2026-06-16 re-review verification:
  `rtk git diff --check -- . ':!.DS_Store'` passed with no output.

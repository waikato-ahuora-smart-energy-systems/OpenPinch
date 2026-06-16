# HENS-03 JSON Fixture Conversion and Problem Adapter

## PRD Summary

Convert OpenHENS example inputs into OpenPinch-owned fixtures and build the
private adapter that translates a prepared `PinchProblem` into solver arrays
while equation models still require arrays.

## User Outcome

The future synthesis workflow starts from normal OpenPinch inputs and prepared
problem state. OpenHENS CSV rows no longer define runtime behavior, but the
solver still receives parity-equivalent arrays during migration.

## Scope

- One-time fixture conversion.
- Unit and economics mapping.
- HEN option configuration.
- Private solver-array adapter and axis maps.
- No public runtime CSV import support.
- No solver model execution.

## Plan Context

Read these sections before implementation:

- [OpenPinch Reuse Commitments](../../../OPENHENS_MIGRATION_PLAN.md#openpinch-reuse-commitments)
- [TargetInput Boundary for HEN](../../../OPENHENS_MIGRATION_PLAN.md#targetinput-boundary-for-hen)
- [Canonical Synthesis Problem Contract](../../../OPENHENS_MIGRATION_PLAN.md#canonical-synthesis-problem-contract)
- [OpenHENS Source Disposition](../../../OPENHENS_MIGRATION_PLAN.md#openhens-source-disposition)
- [Phase 3: JSON Fixture Conversion, Problem Adapter, and Unit Bridge](../../../OPENHENS_MIGRATION_PLAN.md#phase-3-json-fixture-conversion-problem-adapter-and-unit-bridge)

Settled decisions for this task:

- Runtime HEN synthesis must not read source OpenHENS CSVs. Convert source CSV
  examples once into OpenPinch-compatible JSON fixtures.
- Both Four-stream and Nine-stream examples must be recreated in the expected
  OpenPinch-compatible fixture/result formats.
- Four-stream is the baseline case used for routine adapter and later solver
  benchmark gates; Nine-stream remains a final-verification case.
- The adapter must derive private solver arrays from a prepared
  `PinchProblem` execution zone, not raw fixture rows, raw `TargetInput`, a HEN
  schema, an independent cached array payload, or any separate solver-input DTO.
- HEN controls belong in `TargetInput.options` and `CONFIG_FIELD_SPECS`.
- Utility prices belong in `UtilitySchema.price`; shared capital-cost inputs
  belong in existing or generalized OpenPinch costing configuration.
- Kelvin source temperatures must be explicit in converted payloads.
- The required workflow in `README.md` is mandatory; the adapter exists only
  behind the prepared `PinchProblem` path in that workflow.

## Requirements Checklist

- [x] Use the case acceptance matrix in `README.md` as the source of required
      fixture scope. Do not choose additional cases unless HENS-00/HENS-11 first
      names their exact source paths, tiers, grids, and thresholds.
- [x] Recreate
      `Four-stream-Yee-and-Grossmann-1990-1` as an OpenPinch-compatible fixture.
- [x] Recreate
      `Nine-stream-Linnhoff-and-Ahmad-1999-1` as an OpenPinch-compatible fixture.
- [x] Convert each matrix-required case into standard OpenPinch JSON using
      `TargetInput`.
- [x] Put process stream data in `StreamSchema`.
- [x] Put utility data in `UtilitySchema`.
- [x] Put hot utility and cold utility operating costs into
      `UtilitySchema.price`.
- [x] Put HEN search, method, solver, tolerance, output, run id, and best-count
      controls into `TargetInput.options`.
- [x] Add or update `CONFIG_FIELD_SPECS` before any converted fixture relies on
      a new HEN option.
- [x] Map exchanger capital-cost coefficients to existing OpenPinch costing
      configuration where possible.
- [ ] If per-exchanger-kind capital-cost coefficients are needed, extend the
      general OpenPinch costing configuration/schema instead of adding a HEN-only
      economics object.
- [x] Audit OpenHENS process stream cost fields `h_cost` and `c_cost`.
- [ ] If process stream costs affect the algorithm, add a general
      OpenPinch-owned representation such as `StreamSchema.price` or another
      existing costing path.
- [x] If process stream costs do not affect the algorithm, document the evidence
      and keep them out of the runtime fixture schema.
- [x] Require explicit units for Kelvin source temperatures. Do not rely on
      OpenPinch default temperature units.
- [x] Load every converted JSON through `PinchProblem`.
- [x] Confirm `PinchProblem.load(...)` and `prepare_problem(...)` create a
      prepared execution `Zone`, `StreamCollection`, and `Stream` objects.
- [x] Create private `problem_to_solver_arrays(problem, dTmin)` under the
      synthesis service package.
- [x] Ensure the adapter derives arrays only from the live prepared `Zone` owned
      by the `PinchProblem` being solved, or from an immutable prepared-zone
      snapshot created by `prepare_problem(...)` and stored on that same
      `PinchProblem`.
- [x] Preserve source OpenHENS behavior where missing temperature contributions
      become `dTmin / 2` inside the private adapter.
- [x] Include HEN option values, utility prices, costing coefficients, stage
      selection, and solver controls in the adapter output where the moved
      equation models expect them.
- [x] Store labelled axis maps with the adapter output:
      hot process stream key -> `i`, cold process stream key -> `j`,
      stage -> `k`, hot utility key -> utility index, and cold utility key ->
      utility index.
- [x] Add adapter snapshot tests comparing migrated Four-stream arrays to source
      OpenHENS solver-array payloads for the same case and `dTmin`.
- [x] Add adapter snapshot artifacts or exact extraction commands before
      marking adapter work complete. Snapshots must include array shapes, axis
      maps, unit conventions, stream identities, utility identities, and covered
      `dTmin` values.
- [x] Add fixture validation tests for the recreated Nine-stream example, but do
      not require it as the routine adapter baseline unless the task explicitly
      needs final verification.
- [x] Add tests that fail if arrays are built directly from converted fixture
      rows, raw `TargetInput`, or HEN schemas while bypassing `PinchProblem`.
- [x] Add tests that fail if arrays are built from independent cached array
      payloads, standalone solver-input DTOs, or snapshots not stored on the
      same `PinchProblem`.
- [x] Add row and field context to conversion and validation errors.
- [x] Keep conversion tooling out of the public runtime API.

## General Standards That Apply

- [x] Enforce the required OpenPinch workflow from `README.md`; deviations are
      not permitted.
- [x] Runtime synthesis starts from JSON, `TargetInput`, or `PinchProblem`, not
      CSV.
- [x] Public entry points do not accept raw stream lists, raw utility lists,
      CSV rows, or `TargetInput` directly for synthesis.
- [x] Private arrays are a migration bridge and should shrink as solver models
      become OpenPinch-native.
- [x] Fixture conversion must preserve current OpenHENS scientific behavior
      before later replacement work starts.

## Verification Checklist

- [x] Converted fixture validation tests pass for every matrix-required
      OpenHENS case.
- [x] Adapter snapshots match source OpenHENS arrays for each required case and
      `dTmin`.
- [x] Unit conversion tests cover Kelvin source inputs.
- [x] Negative bypass tests prove adapter construction requires a prepared
      `PinchProblem`.
- [x] Bad row values fail with row and field context.

## Definition of Done

- [ ] Four-stream and Nine-stream both exist as OpenPinch-compatible JSON
      fixtures.
- [ ] The public fixture path flows through `TargetInput`, `PinchProblem`,
      `prepare_problem(...)`, `Zone`, `StreamCollection`, and `Stream`.
- [ ] Four-stream private solver-array snapshots are parity-equivalent to
      OpenHENS source arrays.
- [ ] Utility and costing inputs are owned by OpenPinch schemas/configuration.
- [ ] No runtime CSV synthesis API was added.

## Out of Scope

- Moving equation models.
- Running GEKKO/Pyomo solvers.
- Adding visualization.
- Supporting multi-zone HEN synthesis.

## Implementation Notes

- 2026-06-16: Implemented the HENS-03 slice with the README acceptance matrix
  scope only: `Four-stream-Yee-and-Grossmann-1990-1` and
  `Nine-stream-Linnhoff-and-Ahmad-1999-1`.
- Converted fixtures live at
  `tests/fixtures/openhens/Four-stream-Yee-and-Grossmann-1990-1.json` and
  `tests/fixtures/openhens/Nine-stream-Linnhoff-and-Ahmad-1999-1.json`, with
  reordered variants at the HENS-00 paths. Payloads validate as `TargetInput`,
  use `StreamSchema` and `UtilitySchema`, carry explicit `K` temperature units,
  put utility costs in `UtilitySchema.price`, and put HEN controls plus shared
  exchanger costing in `TargetInput.options` / existing `CONFIG_FIELD_SPECS`.
- Added dev-only conversion tooling at `scripts/convert_openhens_fixtures.py`.
  The generation command used was
  `rtk uv run python scripts/convert_openhens_fixtures.py --openhens-root /Users/ca107/Desktop/ahuora/OpenHENS --write`.
  The script is outside the `OpenPinch` package namespace and is covered by a
  negative public-runtime API test.
- Added the private adapter
  `OpenPinch/services/heat_exchanger_network_synthesis/array_adapter.py`.
  `problem_to_solver_arrays(problem, dTmin)` accepts only a prepared
  `PinchProblem`, reads arrays from the live prepared `Zone` stream collections,
  returns labelled axis maps and solver-array metadata, and falls back to
  `dTmin / 2` when prepared temperature contributions are absent.
- Added HENS-00 structural snapshots:
  `openhens_baseline_results/fixture_snapshots/Four-stream-Yee-and-Grossmann-1990-1.json`
  and
  `openhens_baseline_results/fixture_snapshots/Nine-stream-Linnhoff-and-Ahmad-1999-1.json`.
  Added the routine adapter snapshot
  `openhens_baseline_results/adapter_snapshots/Four-stream-Yee-and-Grossmann-1990-1/dTmin-14.json`.
- Process stream cost audit: source OpenHENS emits `h_cost` and `c_cost` in
  `CaseStudy.to_legacy_arrays(...)` and initializes blank model attributes in
  `GenericHENModel`, but active PDM/stage-wise objective, post-processing, and
  verification equations use `hu_cost`, `cu_cost`, and exchanger area/fixed
  costs. No active equation references `h_cost` or `c_cost`, so process stream
  costs remain out of the runtime fixture schema for this slice. The conditional
  checklist item for adding a general `StreamSchema.price` remains unchecked as
  not applicable.
- Per-exchanger-kind capital-cost schema extension remains unchecked as not
  applicable: both matrix-required cases use equal exchange/heating/cooling
  fixed cost, area coefficient, and exponent, so the fixtures map to existing
  `FIXED_COST`, `VARIABLE_COST`, and `COST_EXP`.
- Verification:
  `rtk uv run pytest tests/test_heat_exchanger_network_array_adapter.py tests/test_lib/test_synthesis_schemas.py tests/test_analysis/test_data_preparation.py`
  passed with 101 tests and 3 existing data-preparation warnings.
  `rtk uv run ruff check .` passed.
  `rtk git diff --check -- . ':!.DS_Store'` passed.
- Definition of Done checkboxes are intentionally left unchecked pending
  adversarial review.

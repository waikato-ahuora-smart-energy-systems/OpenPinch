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

- [ ] Use the case acceptance matrix in `README.md` as the source of required
      fixture scope. Do not choose additional cases unless HENS-00/HENS-11 first
      names their exact source paths, tiers, grids, and thresholds.
- [ ] Recreate
      `Four-stream-Yee-and-Grossmann-1990-1` as an OpenPinch-compatible fixture.
- [ ] Recreate
      `Nine-stream-Linnhoff-and-Ahmad-1999-1` as an OpenPinch-compatible fixture.
- [ ] Convert each matrix-required case into standard OpenPinch JSON using
      `TargetInput`.
- [ ] Put process stream data in `StreamSchema`.
- [ ] Put utility data in `UtilitySchema`.
- [ ] Put hot utility and cold utility operating costs into
      `UtilitySchema.price`.
- [ ] Put HEN search, method, solver, tolerance, output, run id, and best-count
      controls into `TargetInput.options`.
- [ ] Add or update `CONFIG_FIELD_SPECS` before any converted fixture relies on
      a new HEN option.
- [ ] Map exchanger capital-cost coefficients to existing OpenPinch costing
      configuration where possible.
- [ ] If per-exchanger-kind capital-cost coefficients are needed, extend the
      general OpenPinch costing configuration/schema instead of adding a HEN-only
      economics object.
- [ ] Audit OpenHENS process stream cost fields `h_cost` and `c_cost`.
- [ ] If process stream costs affect the algorithm, add a general
      OpenPinch-owned representation such as `StreamSchema.price` or another
      existing costing path.
- [ ] If process stream costs do not affect the algorithm, document the evidence
      and keep them out of the runtime fixture schema.
- [ ] Require explicit units for Kelvin source temperatures. Do not rely on
      OpenPinch default temperature units.
- [ ] Load every converted JSON through `PinchProblem`.
- [ ] Confirm `PinchProblem.load(...)` and `prepare_problem(...)` create a
      prepared execution `Zone`, `StreamCollection`, and `Stream` objects.
- [ ] Create private `problem_to_solver_arrays(problem, dTmin)` under the
      synthesis service package.
- [ ] Ensure the adapter derives arrays only from the live prepared `Zone` owned
      by the `PinchProblem` being solved, or from an immutable prepared-zone
      snapshot created by `prepare_problem(...)` and stored on that same
      `PinchProblem`.
- [ ] Preserve source OpenHENS behavior where missing temperature contributions
      become `dTmin / 2` inside the private adapter.
- [ ] Include HEN option values, utility prices, costing coefficients, stage
      selection, and solver controls in the adapter output where the moved
      equation models expect them.
- [ ] Store labelled axis maps with the adapter output:
      hot process stream key -> `i`, cold process stream key -> `j`,
      stage -> `k`, hot utility key -> utility index, and cold utility key ->
      utility index.
- [ ] Add adapter snapshot tests comparing migrated Four-stream arrays to source
      OpenHENS solver-array payloads for the same case and `dTmin`.
- [ ] Add adapter snapshot artifacts or exact extraction commands before
      marking adapter work complete. Snapshots must include array shapes, axis
      maps, unit conventions, stream identities, utility identities, and covered
      `dTmin` values.
- [ ] Add fixture validation tests for the recreated Nine-stream example, but do
      not require it as the routine adapter baseline unless the task explicitly
      needs final verification.
- [ ] Add tests that fail if arrays are built directly from converted fixture
      rows, raw `TargetInput`, or HEN schemas while bypassing `PinchProblem`.
- [ ] Add tests that fail if arrays are built from independent cached array
      payloads, standalone solver-input DTOs, or snapshots not stored on the
      same `PinchProblem`.
- [ ] Add row and field context to conversion and validation errors.
- [ ] Keep conversion tooling out of the public runtime API.

## General Standards That Apply

- [ ] Enforce the required OpenPinch workflow from `README.md`; deviations are
      not permitted.
- [ ] Runtime synthesis starts from JSON, `TargetInput`, or `PinchProblem`, not
      CSV.
- [ ] Public entry points do not accept raw stream lists, raw utility lists,
      CSV rows, or `TargetInput` directly for synthesis.
- [ ] Private arrays are a migration bridge and should shrink as solver models
      become OpenPinch-native.
- [ ] Fixture conversion must preserve current OpenHENS scientific behavior
      before later replacement work starts.

## Verification Checklist

- [ ] Converted fixture validation tests pass for every matrix-required
      OpenHENS case.
- [ ] Adapter snapshots match source OpenHENS arrays for each required case and
      `dTmin`.
- [ ] Unit conversion tests cover Kelvin source inputs.
- [ ] Negative bypass tests prove adapter construction requires a prepared
      `PinchProblem`.
- [ ] Bad row values fail with row and field context.

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

- 

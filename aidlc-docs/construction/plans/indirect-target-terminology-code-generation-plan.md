# Indirect Target Terminology Code Generation Plan

This checklist is the single source of truth for the approved implementation.

## Unit Context

- **Unit**: Indirect-target terminology and reporting metadata.
- **Dependencies**: Existing Zone hierarchy, direct/indirect targeting,
  advanced analysis services, reporting contracts, and graph serialization.
- **Inputs**: Solved Zone targets and immediate-subzone Direct Integration
  targets.
- **Outputs**: Generic indirect runtime targets and explicit public summary
  metadata.
- **Compatibility**: Clean Python API break; historical workbook labels remain
  accepted at the adapter boundary.

## Execution Steps

### Step 1 - Domain and Indirect Analysis Rename

- [x] Add regression-first model and enum expectations.
- [x] Add `SubzoneAggregateTarget` and `IndirectIntegrationTarget`.
- [x] Replace TZ/TS/RT with SA/II and rename the analysis module.
- [x] Rename the subzone-aggregate helper and update target construction.
- [x] Mark Step 1 complete after focused domain and targeting tests pass.

### Step 2 - Application and Advanced Analysis Migration

- [x] Update targeting services, accessors, traversal return selection, and
  Site-only Total Site convenience validation.
- [x] Update HPR, refrigeration, exergy, cogeneration, power, energy transfer,
  multi-period preparation, and graph metadata references.
- [x] Verify generic indirect behavior for all eligible Zone types.
- [x] Mark Step 2 complete after orchestration tests pass.

### Step 3 - Reporting Metadata Contract

- [x] Add Scope, Zone Type, Integration Type, and Target Method metadata.
- [x] Exclude internal aggregates and remove record names from public results.
- [x] Replace summary, metric, Excel, multi-period, and comparison identity.
- [x] Cover canonical nested scopes and target-method disambiguation.
- [x] Mark Step 3 complete after contract and reporting tests pass.

### Step 4 - Boundary, Notebook, and Documentation Migration

- [x] Normalize legacy workbook labels into the new representation.
- [x] Update notebooks to select and display the new metadata columns.
- [x] Update public documentation and API references.
- [x] Remove all stale production and test references to the retired symbols.
- [x] Mark Step 4 complete after adapter, notebook, and documentation checks.

### Step 5 - Focused Verification

- [x] Run domain, targeting, hierarchy, orchestration, reporting, graph,
  multi-period, HPR, energy-transfer, exergy, power, adapter, and workspace
  tests.
- [x] Execute Notebook 2 and verify target and LPS-ledge regressions.
- [x] Run Ruff on all changed Python files.
- [x] Mark Step 5 complete after all focused gates pass.

### Step 6 - Integrated Build and Test

- [x] Run the complete fixed-seed non-solver suite.
- [x] Run repository Ruff lint and format checks.
- [x] Build documentation and distributions and run installed-wheel smoke.
- [x] Run patch-hygiene and stale-symbol checks.
- [x] Generate implementation and Build and Test summaries.
- [x] Mark all remaining workflow and stage checkboxes complete.

## Partial PBT Compliance

- Serialization round trips must preserve the four public metadata fields.
- Weighted-period grouping must be invariant to numeric target values.
- Use the repository Hypothesis seed `20260715` and retain shrinking.

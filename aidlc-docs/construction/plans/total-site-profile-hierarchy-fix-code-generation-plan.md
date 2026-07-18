# Total Site Profile Hierarchy Fix Code Generation Plan

This plan is the single source of truth for the approved implementation unit.

## Unit Context

- **Unit**: Total Site targeting-state isolation.
- **Inputs**: Immediate subzone Direct Integration targets and selected period.
- **Outputs**: Fresh net hot/cold segment collections persisted as the parent
  Zone's immediate-subzone pair and used by its Total Site calculation.
- **Contract**: Adds the requested Zone profile properties without changing
  existing APIs or graph schemas.
- **Property**: Reconstructed process-curve duties match the corresponding
  immediate-subzone utility duties within documented numerical tolerance,
  independent of mutable child net-profile state.

## Execution Steps

### Step 1 - Baseline and Regression-First Checkpoint

- [x] Run focused targeting and graph baselines.
- [x] Add a three-level poisoned-state regression.
- [x] Add Notebook 2 duty, idempotence, and state-ownership regressions.
- [x] Add a bounded Hypothesis invariant with fixed-seed reproducibility.
- [x] Confirm the intended regressions fail before the production correction.

### Step 2 - Deterministic Profile Reconstruction

- [x] Add a helper that reconstructs profiles from immediate child DI targets.
- [x] Preserve selected-period context and stable qualified stream keys.
- [x] Use reconstructed profiles in Total Site process-side cascade creation.
- [x] Retain the no-child `zone.net_*` fallback.
- [x] Mark Step 2 complete immediately after focused implementation checks pass.

### Step 3 - Orchestration State Isolation

- [x] Remove normal indirect targeting's mutating child-profile import.
- [x] Remove the equivalent multi-period indirect HPR mutation.
- [x] Verify `zone.net_*` continues to represent the zone's own direct profiles.
- [x] Mark Step 3 complete immediately after orchestration checks pass.

### Step 4 - Focused Verification

- [x] Run total-site, direct-targeting, graph, service-orchestration,
  multi-scale, multi-period, and HPR preparation tests.
- [x] Execute Notebook 2 and verify the corrected numeric duties.
- [x] Run Ruff on all changed Python files.
- [x] Mark Step 4 complete after all focused gates pass.

### Step 5 - Integrated Build and Test

- [x] Run the complete fixed-seed non-solver suite.
- [x] Run notebook packaging/resource checks and distribution smoke as required.
- [x] Run repository Ruff lint and format checks.
- [x] Generate the implementation and Build and Test summaries.
- [x] Mark every remaining workflow and stage checkbox complete.

### Step 6 - Explicit Per-Zone Profile Ownership

- [x] Add failing domain tests for separate own-zone and immediate-subzone pairs.
- [x] Add failing period-context and immediate-child import tests.
- [x] Add failing Total Site persistence and poisoned-state regressions.
- [x] Add the second hot/cold pair and explicit Zone properties.
- [x] Populate the second pair from immediate-child Direct Integration targets.
- [x] Preserve the own-zone pair, no-child fallback, and graph/API behavior.
- [x] Run focused and complete verification and update final evidence.
- [x] Mark the reopened workflow complete.

### Step 7 - Preserve Rounded SUGCC Utility Ledges

- [x] Add a failing generic duplicate-coordinate corner regression.
- [x] Add a failing `pulp_mill.json` HPS/LPS SUGCC geometry regression.
- [x] Remove consecutive identical coordinate pairs before collinearity cleanup.
- [x] Preserve existing composite-curve edge and segmentation behavior.
- [x] Run focused graph, Total Site, Notebook 2, and Ruff checks.
- [x] Run complete Build and Test gates and update final evidence.
- [x] Mark the reopened workflow complete.

## Partial PBT Compliance Plan

- PBT-02: N/A; no inverse transformation exists.
- PBT-03: Cover duty conservation and poisoned-state independence.
- PBT-07: Use constrained, structured subzone target/profile inputs.
- PBT-08: Retain Hypothesis shrinking and use repository seed `20260715`.
- PBT-09: Use the repository's existing Hypothesis and pytest stack.

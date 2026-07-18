# Residual Compatibility Shim Cleanup Code Generation Plan

This checklist is the single source of truth for the approved cleanup implementation. Every completed step must be marked `[x]` in the same interaction in which its work is completed.

## Unit Context

- **Unit**: Residual compatibility shim cleanup.
- **Stories**: User Stories skipped; traceability is to FR-1 through FR-6 in the approved requirements.
- **Application location**: Existing files under the workspace root.
- **Documentation location**: Existing RTD files under `docs/`; AI-DLC summaries under `aidlc-docs/construction/residual-compatibility-shim-cleanup/`.
- **Dependencies**: NumPy, Hypothesis, Pyomo 6.10 or newer, Sphinx, Ruff, and the optional external OpenHENS checkout used only by the comparison script.
- **Public boundary**: The root package continues exporting only `PinchProblem` and `PinchWorkspace`.
- **Persistence boundary**: No JSON, workspace-bundle, or HEN transport schema changes.
- **Infrastructure and database ownership**: None.

## Planning and Approval

- [x] Read the approved requirements, clarification answers, execution plan, state, reverse-engineering structure, and dependency context.
- [x] Confirm the brownfield target files exist and will be modified in place.
- [x] Map FR-1 through FR-6 to the ten steps below.
- [x] Include example-based, property-based, documentation, architecture, optional-solver, and packaging verification.
- [x] Record the user's explicit approval through task completion as approval for this dependency-ordered plan.

## Step 1: Baseline and Exact Contract Inventory

- [x] Run focused baseline tests for numerics, unit policy, HEN backend, architecture, and documentation consistency.
- [x] Capture exact current callers and test expectations for penalty selection, unit-group overrides, Pyomo availability, the OpenHENS patch, and RTD navigation.
- [x] Confirm unrelated working-tree changes are limited to the current AI-DLC artifacts before application edits.

**Traceability**: FR-1 through FR-6; repository-safety NFR.

## Step 2: Canonical Penalty Enum and Runtime Migration

- [x] Add `PenaltyForm` to `OpenPinch/domain/enums.py` and its concrete-module `__all__` list without changing root exports.
- [x] Change `g_ineq_penalty()` to accept enum values only and dispatch by enum identity.
- [x] Migrate every production caller from string selection to `PenaltyForm`.
- [x] Preserve the square and square-root-smoothing equations exactly.

**Traceability**: FR-1.

## Step 3: Penalty Contract and Property Tests

- [x] Update example tests for both enum members, the default, and rejection of strings and unsupported types.
- [x] Add Hypothesis invariants using finite bounded scalar and one-dimensional residual strategies.
- [x] Use seed `20260715` and retain shrinking.
- [x] Add a root-export guard proving `PenaltyForm` is not exported from `OpenPinch`.

**Traceability**: FR-1; PBT-03, PBT-07, PBT-08, PBT-09.

## Step 4: Unit-Group Terminology Canonicalization

- [x] Rename `InputUnitRule.aliases` and `OutputUnitRule.aliases` to canonical unit-group terminology.
- [x] Rename `_resolve_override()` parameters and all rule construction and use sites.
- [x] Update tests and descriptions without adding transitional constructor parameters or properties.
- [x] Add or retain invariant coverage proving one dimensional override applies to every intended field.

**Traceability**: FR-4 and FR-6; PBT-03 and PBT-07.

## Step 5: Current Pyomo API Enforcement

- [x] Remove the positional `available(False)` retry and call only `available(exception_flag=False)`.
- [x] Add a focused test proving one keyword call is made.
- [x] Add a focused test proving signature `TypeError` propagates without retry.
- [x] Preserve missing-solver and missing-Couenne behavior otherwise unchanged.

**Traceability**: FR-3 and FR-6.

## Step 6: Strict OpenHENS Comparison Prerequisite

- [x] Remove `_install_openhens_compatibility()` and imported-module assignments.
- [x] Add a read-only prerequisite validator for the exact upstream comparison capabilities.
- [x] Ensure unsupported checkouts fail before comparison output directories or tasks are created.
- [x] Add focused tests using isolated fake modules or capability objects, including proof that no upstream objects are mutated.

**Traceability**: FR-2.

## Step 7: RTD and Stale-Terminology Cleanup

- [x] Delete `docs/reference/api-lib.rst` and remove its generated-index entry.
- [x] Rename the stale utility-validation test.
- [x] Add documentation and AST/source guards for the removed page, penalty string aliases, OpenHENS monkeypatch, Pyomo retry, and unit-policy alias terminology.
- [x] Preserve accurate historical release-note statements and intentional normalization documentation.

**Traceability**: FR-5 and FR-6.

## Step 8: Focused Verification and Corrections

- [x] Run numerical, HPR, unit, input, HEN backend, comparison-tool, architecture, and documentation-consistency tests.
- [x] Run fixed-seed Hypothesis properties and confirm shrinking remains enabled.
- [x] Run Ruff lint and format checks on changed Python files.
- [x] Correct all failures and immediately update completed checkboxes.

**Traceability**: All functional and test requirements.

## Step 9: Broad Build and Test Gates

- [x] Run the complete applicable non-solver suite.
- [x] Run the relevant optional HEN solver profile and verify Couenne resilience tests.
- [x] Run warning-free Sphinx validation and confirm deleted navigation is absent.
- [x] Run distribution build, isolated import smoke, stale-symbol scan, and patch-hygiene checks in proportion to affected surfaces.

**Traceability**: Acceptance criteria and all NFRs.

## Step 10: Completion Artifacts and State

- [x] Create the code implementation summary under `aidlc-docs/construction/residual-compatibility-shim-cleanup/code/`.
- [x] Create build, unit-test, integration-test, performance-test, and summary artifacts under the unit build-and-test directory.
- [x] Record exact verification results, extension compliance, and any N/A gates.
- [x] Mark every plan checkbox and AI-DLC stage checkbox complete.
- [x] Append completion evidence to `audit.md` without overwriting prior history.

**Traceability**: AI-DLC completion and audit requirements.

## Completion Criteria

- [x] All ten steps and every nested checkbox are marked complete.
- [x] All acceptance criteria in the approved requirements are satisfied.
- [x] Partial PBT compliance is recorded with no blocking findings.
- [x] No unrelated user changes are overwritten.

# Indirect Profile Precision Code Generation Plan

This checklist is the execution source of truth for the approved fix.

## Step 1 - Regression First

- [x] Add a direct graph regression proving graph extraction does not mutate
  full-precision Problem Tables while graph payloads remain rounded.
- [x] Add an indirect reconstruction regression with temperature and enthalpy
  values requiring more than four decimal places.
- [x] Run the focused tests and confirm the regression fails before the fix.

## Step 2 - Precision-Preserving Implementation

- [x] Round graph-only Problem Table copies in direct targeting.
- [x] Remove four-decimal rounding from reconstructed indirect segments.
- [x] Preserve all existing APIs and graph schemas.

## Step 3 - Focused Verification

- [x] Run direct, indirect, hierarchy, graph, multi-period, and HPR tests.
- [x] Run Ruff lint and format checks on changed Python files.
- [x] Update the implementation summary and same-interaction checkboxes.

## Step 4 - Integrated Build and Test

- [x] Run the complete fixed-seed non-solver suite.
- [x] Run Notebook 2 and packaging/resource verification.
- [x] Run repository Ruff, format, and patch-hygiene checks.
- [x] Build the distribution and run an isolated wheel smoke.
- [x] Generate Build and Test records and complete workflow state.

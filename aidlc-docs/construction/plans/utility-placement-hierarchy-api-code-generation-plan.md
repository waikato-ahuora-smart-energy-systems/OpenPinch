# Utility Placement Hierarchy API Code Generation Plan

## TDD Execution Steps

- [x] **Step 1 - Requirements and plan.** Validate all hierarchy and count
  answers, update the comprehensive requirements, and approve this bounded
  plan under the user's continuing authorization.
- [x] **Step 2 - RED: public signature and routing.** Require optional counts,
  optional zone selection, no public `base_target` or template collections,
  master-zone defaulting, and type-derived direct/Total Site/indirect routing.
- [x] **Step 3 - GREEN: zone resolution and scope inference.** Add detached,
  unique hierarchy resolution with typed missing, ambiguous, foreign, and
  unsupported-zone failures; add aggregate-indirect scope where required.
- [x] **Step 4 - RED: existing-utility inference.** Require thermal-kind
  classification, missing-target isothermal handling, `Both` expansion,
  opposite temperature direction, deterministic padding, minimum inferred
  inventory, and explicit-count override.
- [x] **Step 5 - GREEN: application template inference.** Convert validated
  input utilities to specialist templates without mutating source data and
  preserve supported identities and metadata.
- [x] **Step 6 - RED-GREEN: batch and all-period surfaces.** Mirror the same
  shared-placement behavior across ordered cases and canonical periods.
- [x] **Step 7 - RED-GREEN: notebook, RTD, and inventory.** Demonstrate separate
  Process/GCC and Site/TSP workflows, remove public `base_target`, and keep one
  executable notebook plus exact operation coverage.
- [x] **Step 8 - Properties and refactor.** Verify deterministic classification,
  `Both` pairing, padding, declaration-order invariance, detached copying,
  hierarchy routing, and numerical-regression preservation.
- [x] **Step 9 - Focused and complete gates.** Run specialist and application
  tests, Ruff, patch hygiene, warning-clean RTD, full solver-enabled tests,
  fresh distributions, and installed-wheel notebook execution.
- [x] **Step 10 - Completion records.** Review every changed file for necessity
  and update summaries, state, audit, and checkboxes with final evidence.

## Production Paths

- `OpenPinch/contracts/utility_placement.py`
- `OpenPinch/analysis/utility_placement/normalization.py`
- `OpenPinch/application/utility_placement.py`
- `OpenPinch/application/_problem/accessors/target.py`
- `OpenPinch/application/workspace.py`
- `scripts/generate_tutorial_notebooks.py`
- notebook 19, RTD API/guide pages, and tutorial inventory

## Test Paths

- `tests/application/test_utility_placement.py`
- `tests/application/test_utility_placement_batch.py`
- `tests/analysis/utility_placement/`
- `tests/packaging/test_notebooks.py`
- `tests/packaging/test_docs_consistency.py`
- package usability, architecture, resource, and tutorial inventory gates

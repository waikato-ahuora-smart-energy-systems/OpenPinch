# Utility Placement Profile Envelope Correction Code Generation Plan

## Unit Context

- **Unit**: Placement Evaluation and Optimisation Service with public notebook
  integration.
- **Defect**: Generated utility supply bounds sit outside the complete process
  temperature envelope and sensible spans cannot cover the residual-profile
  support. The optimizer therefore cannot minimize the visible profile gap.
- **Dependencies**: Existing detached context, targeting allocation, balanced-
  composite entropy kernel, case-based API, standard plots, and notebook 19.
- **Scope**: Numerical bound/start correction plus regression evidence; no API,
  monetary, CLI, plotting, or infrastructure change.

## TDD Execution Steps

- [x] **Step 1 - Reproduce and localize.** Execute Process and Site notebook
  workflows, inspect utilities, target profiles, graph coordinates, and model
  bounds, and identify the unreachable residual-profile temperature support.
- [x] **Step 2 - RED: profile-envelope regression.** Add explicit Process and
  Site examples proving supply bounds and sensible spans cover the residual
  profile and that returned active utilities cover its temperature support.
- [x] **Step 3 - GREEN: support-aware physical envelope.** Derive candidate
  bounds and spans from the residual hot/cold profile support rather than
  placing every level outside the global process extremes.
- [x] **Step 4 - GREEN: deterministic feasible starts.** Seed hot and cold
  placements at their respective residual-profile edges while preserving
  ordering, separation, bounds, reproducibility, and all-period intersection.
- [x] **Step 5 - Objective and property verification.** Prove closer profile
  coverage ranks ahead of separated coverage, inactive coordinates remain
  irrelevant, bounds are valid under generated profile envelopes, and fixed-
  seed results are deterministic.
- [x] **Step 6 - Notebook and RTD correction.** Add executable notebook
  assertions that reject large Process and Site profile-support gaps and update
  explanatory documentation only where behavior changed.
- [x] **Step 7 - Focused and complete gates.** Run utility-placement,
  application, notebook, documentation, PBT, Ruff, patch-hygiene, Sphinx,
  package, installed-wheel, and full repository verification.
- [x] **Step 8 - Completion records.** Review change necessity and update the
  code summary, Build and Test summary, state, audit, and this checklist with
  measured before/after evidence.

## Production Paths

- `OpenPinch/application/utility_placement.py`
- `OpenPinch/analysis/utility_placement/bounds.py`
- `OpenPinch/analysis/utility_placement/codec.py` only if start construction
  requires correction
- `scripts/generate_tutorial_notebooks.py`
- notebook 19 and its RTD guide only when required by the corrected behavior

## Test Paths

- `tests/application/test_utility_placement.py`
- `tests/analysis/utility_placement/test_bounds.py`
- `tests/analysis/utility_placement/test_properties.py`
- `tests/packaging/test_notebooks.py`

The user's standing authorization approves this bounded correction through
completion unless an unexpected scope or compatibility issue appears. Full
Property-Based Testing remains enabled; Security and Resiliency remain
disabled and N/A.

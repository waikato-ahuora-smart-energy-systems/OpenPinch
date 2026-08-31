# Utility Placement Coupled Generated Pairs Code Generation Plan

## Unit Context

- **Units**: Placement Contracts and Pure Model; Placement Evaluation and
  Optimisation Service; Public Workflow and Notebook Integration.
- **Approved rule**: Every matching generated hot/cold pair of the same kind
  and ordinal shares one temperature interval. Cold endpoints exactly reverse
  hot endpoints; duties remain independent.
- **Compatibility**: Unrelated inferred or explicit Hot/Cold templates remain
  independent. No monetary, CLI, plotting-surface, or infrastructure change.
- **Properties**: exact endpoint inversion, reduced generated-vector dimension,
  encode/decode round trip, pair ordering, bounds, feasible starts, independent
  duty conservation, deterministic optimization, and source isolation.

## TDD Execution Steps

- [x] **Step 1 - Record the approved contract.** Update requirements and
  functional rules for exact generated-pair reversal, shared coordinates,
  independent duties, and pair ordering.
- [x] **Step 2 - RED: pair contracts and properties.** Add example and
  Hypothesis regressions for generated isothermal/sensible inversion, reduced
  coordinate dimension, ordering, round trips, and independent inferred mode.
- [x] **Step 3 - GREEN: generated coordinate schema.** Build one supply
  coordinate per isothermal pair and one supply/span pair per sensible pair;
  retain the existing independent schema for inferred/explicit templates.
- [x] **Step 4 - GREEN: coupled bounds, ordering, and starts.** Use common
  physical support for generated pairs, enforce both endpoint bounds and shared
  pair ordering, and generate independently verified deterministic starts.
- [x] **Step 5 - GREEN: exact codec and verification.** Decode cold members by
  endpoint reversal, reject inconsistent encoded placements, preserve stable
  identities, and keep hot/cold duty allocation independent.
- [x] **Step 6 - Application and Total Site regression.** Prove Process and
  Site optimized cases contain exact paired endpoints, no generated fallback,
  stable source state, and standard-plot-compatible utility definitions.
- [x] **Step 7 - Notebook, requirements, RTD, and summaries.** Add executable
  pair assertions to notebook 19 and update only behaviorally affected docs and
  AI-DLC records.
- [x] **Step 8 - Complete verification.** Run focused/PBT tests, Ruff, patch
  hygiene, Sphinx, packaging, fresh wheel, installed notebook, and the complete
  solver-enabled repository suite; record exact evidence and close state.

## Production Paths

- `OpenPinch/contracts/utility_placement.py`
- `OpenPinch/analysis/utility_placement/bounds.py`
- `OpenPinch/analysis/utility_placement/codec.py`
- `OpenPinch/application/utility_placement.py`
- `scripts/generate_tutorial_notebooks.py`
- `OpenPinch/data/notebooks/19_utility_placement_optimisation.ipynb`

## Test Paths

- `tests/analysis/utility_placement/test_codec.py`
- `tests/analysis/utility_placement/test_bounds.py`
- `tests/analysis/utility_placement/test_properties.py`
- `tests/application/test_utility_placement.py`
- `tests/packaging/test_notebooks.py`

The user's standing authorization and explicit confirmation approve this plan
through completion unless an unexpected compatibility or feasibility conflict
appears. Property-Based Testing is enabled; Security and Resiliency are
disabled and N/A.

# Heat-Recovery `dt_min` Build and Test Summary

## Results

`dt_min` terminology verification:

- Focused contract, analysis, application, and property suite: 47 passed.
- Notebook, documentation, and public-inventory gate: 52 passed, 3 expected
  optional-profile skips.
- Warning-strict Sphinx build: passed across 55 source files.
- Full configured repository suite: 2,584 passed, 4 skipped in 418.36 seconds.
- Exact non-solver coverage selection: 2,581 passed, 3 skipped, 4 deselected
  in 403.68 seconds; 96 percent branch coverage passed the 95 percent gate.
- Ruff lint, feature-file formatting, and patch hygiene: passed.
- Wheel and source distribution: built successfully as OpenPinch 0.6.3.
- Isolated Python 3.14 installed-wheel smoke: passed from site-packages using
  `target.heat_recovery_dt_min(...)` and `result.dt_min`.

Threshold-limit correction verification:

- Inverse solver, contract, application, and property suite: 46 passed.
- Notebook and RTD gate: 47 passed, 3 expected optional-profile skips.
- Full configured repository suite: 2,583 passed, 4 skipped in 407.53 seconds.
- Exact non-solver coverage selection: 2,580 passed, 3 skipped, 4 deselected
  in 313.05 seconds; 96 percent branch coverage passed the 95 percent gate.
- Ruff lint, feature-file formatting, and patch hygiene: passed.

Original feature verification:

- Focused contract, analysis, application, architecture, and documentation:
  134 passed.
- Complete packaged notebook gate: 21 passed, 3 optional-profile skips.
- Final focused configured-input regression: 18 passed.
- Full configured repository suite: 2,578 passed, 4 skipped in 419.05 seconds.
- Statement and branch coverage: 96 percent; the 95 percent gate passed.
- Repository-wide Ruff lint: passed.
- New and materially edited feature-file Ruff formatting: passed.
- Warning-strict Sphinx build, notebook drift, compilation, execution, tutorial
  ownership, and patch hygiene: passed.
- Wheel and source distribution: built successfully as OpenPinch 0.6.3.
- Isolated Python 3.14 installed-wheel smoke: passed from site-packages,
  including the packaged 10 delta-degC forward/inverse regression.

The repository-wide format-only command still identifies 16 pre-existing
files outside this feature; they were not reformatted because that unrelated
baseline rewrite is outside the approved scope.

## Property-Based Testing compliance

- PBT-01 compliant: zero, limit, interior, no-overlap, empty-side, invalid, and
  tolerance boundaries are explicit, including positive threshold-limit
  plateaus.
- PBT-02 compliant: generated sensible, segmented, inactive, and multiperiod
  combinations are exercised.
- PBT-03 compliant: order, units, repetition, forward/inverse, and
  sequential/parallel metamorphic relations are checked.
- PBT-04 compliant: monotonicity, feasibility, bounds, maximality, and
  non-mutation invariants are asserted.
- PBT-05 compliant: reusable domain strategies generate schema-valid process
  cases.
- PBT-06 N/A: the inverse solver owns no persistent mutable state.
- PBT-07 compliant: the packaged forward/inverse case, threshold-limit
  boundary, and an interior plateau endpoint are permanent regressions.
- PBT-08 compliant: properties use fixed seed `20260902`; the full suite also
  uses repository seed `20260715`.
- PBT-09 compliant: standard Hypothesis shrinking remains enabled.
- PBT-10 compliant: all properties run through normal pytest and coverage
  workflows.

Security and Resiliency extensions remain disabled and were not enforced.

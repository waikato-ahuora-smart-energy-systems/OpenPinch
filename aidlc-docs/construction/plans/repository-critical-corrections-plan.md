# Repository Critical Corrections Plan

This plan corrects the six reproduced repository review findings without
changing unrelated numerical methods, public workflow names, notebook content,
or plotting behavior. Tests are written before implementation and the complete
configured-solver suite is the final gate.

## TDD checklist

- [x] **Step 1 - RED problem-state transactions.** Prove failed `load()` and
  failed input replacement preserve the complete previous problem state.
- [x] **Step 2 - RED targeting invariants.** Prove malformed one-dimensional
  utility load profiles fail and documented segmented utilities can be
  retargeted without mutating a derived parent duty directly.
- [x] **Step 3 - RED persistence invariants.** Prove equal selected-period caps
  retain identities and unselected periods remain unbounded, and prove
  `dt_cont_multiplier` survives problem and workspace round trips.
- [x] **Step 4 - RED optimizer feasibility.** Prove every returned optimizer
  candidate satisfies supplied constraints or the solve fails explicitly,
  including incompatible or failed local-polish paths.
- [x] **Step 5 - GREEN state and persistence corrections.** Make problem input
  replacement atomic, persist multiplier changes in canonical zone input, and
  retain maximum-duty period identities for detached cases.
- [x] **Step 6 - GREEN targeting and optimizer corrections.** Count qualifying
  load-profile rows correctly, apply segmented utility duties through their
  authoritative children, and validate constraint feasibility before ranking.
- [x] **Step 7 - Focused and property verification.** Run affected application,
  targeting, optimization, segmented-stream, workspace, utility-placement,
  Ruff, Sphinx, and patch-hygiene gates.
- [x] **Step 8 - Complete verification and records.** Run the configured-solver
  suite without deselection, review changed-file necessity, and align state,
  audit, design, implementation, RTD, and Build and Test evidence.

## Property-Based Testing compliance

- Failed state transitions leave canonical input, prepared zone, cached
  results, period results, and source identity observationally unchanged.
- A multiplier serialized and restored at any valid zone retains its value.
- Period-aware caps retain selected identities regardless of whether their
  magnitudes are equal; omitted periods resolve as unbounded.
- Every optimizer result is finite, within bounds, and constraint-feasible.
- Segmented utility assignment conserves the assigned parent duty across child
  segments and is idempotent for repeated targeting.
- Shrinking remains enabled through the existing Hypothesis configuration and
  explicit reviewed examples remain as permanent regressions.

Security and Resiliency extensions remain disabled. Operations is not affected.

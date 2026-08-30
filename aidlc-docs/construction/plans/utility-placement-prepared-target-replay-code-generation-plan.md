# Utility Placement Prepared Target Replay Code Generation Plan

This plan is the single source of truth for the utility-placement replay
performance correction. The user's direct instruction and continuing approval
authorize the complete TDD sequence.

- [x] **Step 1 - Profile and lifecycle deep dive.** Trace `PinchProblem` load,
  preprocessing, direct targeting, utility targeting, Total Site aggregation,
  and optimizer polishing. Confirm that exact replay dominates candidate cost
  and that process cascades are independent of utility duties.
- [x] **Step 2 - Requirements and design boundary.** Require process-only
  profiles to be prepared once per zone and period, copied per candidate, and
  augmented with utility temperature intervals. Retain candidate-specific
  utility targeting, balanced composites, net Total Site aggregation, and
  detached state.
- [x] **Step 3 - RED direct-profile oracle coverage.** Add example and
  Hypothesis tests comparing prepared-profile replay with fresh direct
  targeting across valid utility temperature sets, including intervals above,
  within, and below process support.
- [x] **Step 4 - GREEN reusable direct targeting profile.** Refactor the direct
  target owner to prepare process-only shifted and real problem tables once,
  deep-copy them, insert utility endpoints with the existing `ProblemTable`
  interval engine, and continue through existing utility and balanced-composite
  methods.
- [x] **Step 5 - RED application replay coverage.** Add Process, Total Site,
  multiperiod, source-isolation, pickle, and construction-count tests proving
  that candidate replay is equivalent to a newly constructed problem while
  preprocessing occurs once per adapter rather than once per candidate.
- [x] **Step 6 - GREEN prepared `PinchProblem` replay.** Prepare one detached
  base problem and direct-profile cache in the application adapter, clone its
  prepared zone state, replace only utility collections, and invoke existing
  direct or indirect target execution with the prepared profiles.
- [x] **Step 7 - Performance verification.** Benchmark uncached candidate
  replay for Process and Total Site against the fresh-problem oracle and record
  construction, targeting, entropy, and end-to-end timings.
- [x] **Step 8 - Focused and broad Build and Test.** Run fixed-seed properties,
  direct/indirect targeting, utility placement, Ruff, patch hygiene,
  documentation, and the complete applicable test suite.
- [ ] **Step 9 - Records, review, and completion.** Update requirements, design,
  state, audit, code summary, and Build and Test evidence; review the diff for
  necessity; preserve unrelated notebook and `.gitignore` edits; commit the
  approved correction to `develop`.

## PBT Compliance Plan

- PBT-01: prepared-versus-fresh equivalence is the primary oracle property.
- PBT-03: source immutability, duty conservation, and candidate isolation are
  invariants.
- PBT-05: fresh ordinary targeting is the independent reference oracle.
- PBT-07: generators constrain utility orientation, absolute temperatures,
  process support, maximum duties, and period structure.
- PBT-08: retain Hypothesis shrinking and the repository fixed seed.
- PBT-09: use the existing Hypothesis and pytest stack.
- PBT-10: pair generated oracle coverage with explicit Process, Total Site,
  and multiperiod regression examples.
- PBT-02, PBT-04, and PBT-06: N/A; the correction introduces no inverse,
  idempotent public operation, or stateful command model.

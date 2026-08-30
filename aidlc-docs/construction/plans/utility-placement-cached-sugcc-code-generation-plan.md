# Utility Placement Cached SUGCC Code Generation Plan

This plan is the single source of truth for replacing per-candidate aggregate
target replay with cached process net-load profiles and candidate SUGCC
targeting. The user's `proceed` instruction and standing approval through
completion authorize this TDD sequence unless an unexpected result appears.

## Unit context

- Unit: utility-placement aggregate-scope candidate allocation.
- Requirements: prepared target replay amendment; FR-008 and FR-009;
  Acceptance 18, 25-38.
- Owners: `OpenPinch/application/utility_placement.py`, shared targeting
  functions under `OpenPinch/analysis/targeting/`, and application tests.
- Dependencies: completed direct process profiles,
  `target_utilities_for_load_profiles`, site utility profile construction,
  same-level utility matching, and balanced-composite entropy evaluation.
- Public API and serialized contracts remain unchanged.

## Execution checklist

- [x] **Step 1 - RED exact-equivalence examples.** Add Total Site and higher-
  aggregate tests comparing cached candidate allocation with fresh ordinary
  targeting by name, fallback, target total, and entropy-relevant process
  profile.
- [x] **Step 2 - RED lifecycle and property coverage.** Add call-count tests
  proving process profiles are completed once and aggregate targeting is not
  invoked per candidate, plus generated endpoint/oracle coverage with
  Hypothesis shrinking and the repository seed.
- [x] **Step 3 - GREEN completed process-profile cache.** Prepare the final
  utility-independent `H_NET_A` and separated load profiles once per immediate
  process zone and period, preserving the shared temperature tolerance and
  existing interpolation behavior.
- [x] **Step 4 - GREEN SUGCC-only aggregate allocation.** Build fresh candidate
  utilities without cloning a problem tree, target each cached process
  profile, aggregate duties, construct one candidate SUGCC, apply same-level
  cancellation, and return a cached invariant process-composite snapshot for
  entropy.
- [x] **Step 5 - REFACTOR and final replay gate.** Keep Direct prepared replay,
  retain ordinary targeting as the independent oracle and winning-case
  acceptance path, remove redundant aggregate candidate work, and preserve
  typed candidate diagnostics and multiperiod isolation.
- [x] **Step 6 - Focused performance and regression verification.** Run the
  exact-equivalence/property tests, aggregate targeting and utility-placement
  suites, measure representative candidate speed, and address every failure.
- [x] **Step 7 - Broad Build and Test and records.** Run applicable complete
  tests, Ruff, format, documentation, and patch hygiene; update requirements,
  state, audit, implementation summary, and Build and Test evidence; review all
  changed files for necessity while preserving the user's notebook and
  `.gitignore` edits.

## PBT compliance plan

- PBT-01: oracle, invariance, conservation, and isolation properties are
  documented above and in the requirements amendment.
- PBT-03: generated candidates preserve process-profile geometry, named-duty
  conservation, and source/candidate isolation.
- PBT-05: fresh ordinary aggregate targeting is the independent oracle.
- PBT-07: generators constrain oriented utility temperatures to the placement
  feasibility envelope and include boundary-adjacent endpoints.
- PBT-08: Hypothesis shrinking remains enabled and failures report the project
  seed.
- PBT-09: the existing pytest/Hypothesis stack is retained.
- PBT-10: fixed Total Site and hierarchy examples complement generated oracle
  checks.
- PBT-02, PBT-04, and PBT-06: N/A; no inverse, idempotent public operation, or
  new stateful command model is introduced.

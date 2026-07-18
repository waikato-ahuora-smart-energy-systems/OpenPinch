# Unit 1 Code Generation Plan

This plan is the single source of truth for Unit 1 generation. It implements
the contract foundation for US-1, US-2, US-4, and US-8 and supports every later
unit. It depends only on the approved application design and existing domain/
report contracts. No database, frontend, deployment service, or repository
layer is introduced.

- [x] **Step 1: Aggregation policy implementation** — modify
  `OpenPinch/application/_problem/periods/aggregation.py` so optional diagnostic
  values tolerate partial absence while required values retain strict errors.
- [x] **Step 2: Example regression coverage** — update
  `tests/application/test_multiperiod_summary.py` with real partially missing
  pinch-temperature behavior and required-field contrast.
- [x] **Step 3: Domain strategy generation** — create
  `tests/strategies/period_outputs.py` for valid aligned period outputs and
  optional diagnostics.
- [x] **Step 4: Property coverage** — create
  `tests/application/test_multiperiod_summary_properties.py` for JSON round
  trips, order/range/non-mutation, and weight-scale invariants.
- [x] **Step 5: Public contract foundation** — create
  `tests/application/test_package_usability_contract.py` for exact root exports
  and target public vocabulary consumed by Units 2 and 3.
- [x] **Step 6: Focused verification** — run Unit 1 aggregation/example/PBT tests
  with seed `20260715`, Ruff on changed Python files, and patch hygiene.
- [x] **Step 7: Generation summary** — create
  `aidlc-docs/construction/contract-correctness-foundation/code/code-summary.md`
  with modified/created files and evidence.

## PBT Generation Requirements

- PBT-02: generated TargetOutput JSON round trip.
- PBT-03: ordering, numeric range, scale invariance, and non-mutation.
- PBT-07: centralized composite period-output strategy.
- PBT-08: normal Hypothesis shrinking and fixed seed in verification.
- PBT-09: existing Hypothesis dependency and pytest integration.

The user's blanket approval authorizes this complete step sequence.

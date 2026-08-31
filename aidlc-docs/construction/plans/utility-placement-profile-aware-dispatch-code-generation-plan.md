# Utility Placement Profile-Aware Dispatch Code Generation Plan

This approved TDD plan is the single source of truth for correcting utility
placement after comparison with the heat-pump optimizer.

## Unit context

- Unit: utility-placement contracts, evaluation/optimization, and public example.
- Dependencies: existing balanced-composite entropy kernel, reusable optimizer,
  utility targeting, detached-case application integration, and standard plots.
- Interfaces: the public `target.utility_placement(...)` call remains unchanged.
- Persistent data: none.
- Story/requirement traceability: candidate duty allocation, thermodynamic cost,
  multiperiod aggregation, fallback penalties, and the profile-aware dispatch
  correctness amendment in `utility-placement-requirements.md`.

## Testable properties and PBT scope

- PBT-02 round trip: valid temperature/duty coordinates decode and encode without
  loss within documented floating-point tolerance.
- PBT-03 invariants: decoded duties are finite and non-negative, sum to each
  period/side residual, respect `[0, 1]` split bounds, and keep reversed paired
  temperatures exact.
- PBT-05 oracle: small generated duty families agree with a direct
  stick-breaking reference implementation. Advisory under the configured partial
  enforcement mode but included when practical.
- PBT-07/PBT-08: reuse domain strategies and Hypothesis shrinking/seed reporting.
- PBT-10: retain explicit analytical and notebook-derived regressions alongside
  properties.
- PBT-04 is N/A: no new idempotent operation. PBT-06 is N/A: evaluation remains
  detached and process-local rather than a business state machine. PBT-09 is
  already satisfied by the documented Hypothesis stack.

## Execution steps

- [x] Step 1 — Add failing analytical and application regressions for explicit
  profile-aware duty dispatch, candidate ranking, entropy scaling, and evaluation
  accounting; demonstrate the intended RED failures.
- [x] Step 2 — Extend immutable contracts and coordinate encoding with
  period-specific bounded hot/cold duty splits, deterministic feasible starts,
  decode/encode round trips, and property-based invariants.
- [x] Step 3 — Replace greedy optimum duty selection with requested-dispatch
  replay against temperature availability, maximum-duty limits, exact coverage,
  and residual fallback evidence.
- [x] Step 4 — Make backend and canonical candidate ranking identical, introduce
  a context-derived entropy-unit normalization reference, and correct termination
  evaluation accounting.
- [x] Step 5 — Add structurally ordered normalized temperature coordinates where
  they can eliminate rejected combinations without changing physical public
  temperatures; retain independent reversed-pair duty behavior.
- [x] Step 6 — Run focused tests and property tests, refactor only with the focused
  gate green, and record any intentionally deferred encoding edge with rationale.
- [x] Step 7 — Update the requirements-owned RTD guide and notebook 19, execute
  both Process and Total Site workflows, and inspect standard GCC/TSP results.
- [x] Step 8 — Run Ruff, patch hygiene, documentation, packaging/notebook gates,
  and the complete solver-enabled test suite; update summaries and state with
  exact results.

## Approval

The user approved the diagnosis and recommended TDD correction sequence with
the exact response `Go` on 2026-08-29. Standing authorization applies through
completion unless an unexpected scope or external blocker appears.

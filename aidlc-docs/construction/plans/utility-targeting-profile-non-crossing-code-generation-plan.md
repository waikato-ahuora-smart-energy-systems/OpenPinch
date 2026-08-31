# Utility Targeting Profile Non-Crossing Code Generation Plan

This approved corrective TDD plan is the single source of truth for preventing
sensible utility profiles from crossing the Process grand composite curve.

## Unit context

- Unit: shared direct utility targeting with utility-placement exact replay.
- Defect: the duty maximizer computes an interior sensible-profile limit but
  can discard it and allocate the endpoint duty, allowing the Utility GCC to
  cross the Process GCC.
- Dependencies: `target_utilities_for_load_profiles`, ordinary direct targeting,
  the exact-target replay adapter, and standard GCC plotting.
- Scope: correct shared targeting and its tests; preserve the public utility-
  placement API, entropy objective, detached replay, and user-modified notebook.

## Testable properties and PBT scope

- PBT-03 invariant: at every temperature breakpoint, the cumulative hot or cold
  utility profile remains within the corresponding residual Process GCC profile
  within numerical tolerance.
- PBT-05 oracle: the maximum sensible duty agrees with the minimum breakpoint
  ratio for a small piecewise-linear profile.
- PBT-07 and PBT-08: use bounded monotone profile generators with normal
  Hypothesis shrinking and seed reporting.
- PBT-10: retain the notebook-derived 97 degC counterexample as an explicit
  regression alongside the generated invariant.
- PBT-02 is N/A because no serialization or inverse operation changes. PBT-04
  is N/A because no idempotent operation changes. PBT-06 is N/A because shared
  utility targeting remains stateless. PBT-09 remains satisfied by Hypothesis.

## Execution steps

- [x] Step 1 - Reproduce and localize the notebook crossing numerically. Confirm
  that the live exact replay reaches `target_utilities_for_load_profiles` and
  that `_maximise_utility_duty` discards its tighter interior limit.
- [x] Step 2 - Add a failing example regression for the notebook-derived hot
  sensible profile and a property-based breakpoint non-crossing invariant.
- [x] Step 3 - Correct shared utility duty maximization so every assigned
  sensible profile respects all active GCC breakpoints, including capacity and
  prior-level accounting.
- [x] Step 4 - Run focused shared-targeting, direct-targeting, utility-placement,
  and application tests; refactor only with the focused gate green.
- [x] Step 5 - Synchronize requirements, RTD, the canonical notebook generator,
  and notebook 19 only where behavior or explanation changes. Do not add tests
  or assertions to the notebook.
- [x] Step 6 - Run broader Ruff, property, documentation, notebook, packaging,
  and repository test gates; review the diff and update completion records.

## Authorization

The correction is covered by the user's standing approval through completion.
The user explicitly identified GCC crossing as a violation that must be
impossible. Security and Resiliency extensions remain disabled; full
Property-Based Testing remains enabled.

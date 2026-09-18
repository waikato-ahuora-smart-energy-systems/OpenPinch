# Effective HEN contribution verification

The user changed arrays.py to read effective_delta_t_contribution and return it
unchanged above tol, with dTmin / 2 only as fallback. The assistant preserved
that implementation and corrected obsolete test expectations. A real zone
multiplier is now covered at the array integration boundary; new examples and
fixed-seed generated segmented streams verify the numerical contract.

Verification: 102 affected tests passed; 28 adapter tests passed; the full HEN
run completed with 459 passed, 1 skipped and two failures subsequently resolved
by the adapter-test correction and an outside-sandbox image-export rerun.
Ruff and diff whitespace checks passed. No package/dependency changes.

PDM follow-up: targeting now snapshots effective contributions for all streams,
utilities and explicit segments before normalizing copied multipliers. It uses
the shared adapter helper, preserving contributions above tol and falling back
only otherwise. Original PinchProblem values, multipliers and locks are preserved.
Decomposition metadata now documents fallback semantics. Repeated preparation
is idempotent, including zero multipliers and multiperiod segment contributions.
The former adapter/PDM policy mismatch is resolved.

Follow-up verification so far: five focused regressions failed before the fix;
all 113 tests across pinch parity, segmented streams and PDM model boundaries
pass after it. Ruff passes. Full HEN suite: 463 passed, 1 skipped in 113.37 seconds (outside the sandbox for Chrome image export).

Extension compliance: PBT-01/03/05/07/08/09/10 compliant (documented properties,
examples, generated valid segment objects, independent domain multiplier oracle,
fixed seed, shrinking and existing Hypothesis dependency). PBT-02 N/A (no inverse),
PBT-04 compliant (generated repeated PDM normalization), PBT-06 N/A (read-only adapter).
Security/Resiliency disabled and skipped. Operations N/A. No commit or push.

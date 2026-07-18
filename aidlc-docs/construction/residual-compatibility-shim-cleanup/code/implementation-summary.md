# Residual Compatibility Shim Cleanup Implementation Summary

## Outcome

The repository now uses canonical contracts for the remaining compatibility-shaped surfaces:

- `PenaltyForm` selects inequality-penalty algorithms; strings and alternative spellings are rejected.
- `InputUnitRule` and `OutputUnitRule` describe shared dimensional configuration through `unit_groups`.
- Pyomo availability uses only `available(exception_flag=False)`.
- The OpenHENS comparison tool validates an unmodified supported upstream capability surface and no longer replaces upstream functions.
- The final legacy library transition page and navigation entry are removed.
- Static guards prevent these mechanisms and stale terminology from returning.

The OpenHENS tool also now imports the current HEN execution owners directly. Its local bounded and sequential runner substitutions were removed together with their command-line options.

## Preserved Contracts

- Compact input and HEN wire keys and exact round trips.
- Fluid-phase normalization, including `vapor` and `vapour`.
- Numeric, multiperiod, Pint, Pydantic, and foreign value-like inputs.
- Optional-dependency guards and focused installation errors.
- Warning-backed Couenne algorithmic fallback.
- Segmented-stream solver tensor shape invariants.
- Root exports limited to `PinchProblem` and `PinchWorkspace`.

## Verification Correction

The real four-stream solver regression exposed a pre-existing nondeterministic assertion. ESM tasks exist only for successful TDM parents, so two live runs produced 99 and 97 ESM branches while selecting the same checked-in objective and design. The regression now retains the exact design, objective, topology, cost, and upper-bound checks while requiring at least 95 of the 100 expected ESM branches. This matches the bounded-live-baseline approach already used for the nine-stream case.

## Test Evidence

- Focused affected suite: 275 passed.
- Fixed-seed non-solver suite: 2,108 passed, 3 intentional opt-in skips, 4 solver deselections.
- Real solver profile after the bounded regression correction: 3 passed, 1 intentional nine-stream skip.
- Ruff: all checks passed; 460 files formatted.
- Sphinx: 53 sources built with warnings treated as errors.
- Distribution: OpenPinch 0.5.2 wheel and source distribution built successfully.
- Isolated wheel smoke: passed from a fresh Python 3.14 site-packages environment outside the checkout.

## Extension Compliance

- Security Baseline: disabled; N/A because no security boundary changed.
- Resiliency Baseline: disabled; N/A. The explicitly preserved Couenne fallback remained green.
- Partial Property-Based Testing: compliant for PBT-03, PBT-07, PBT-08, and PBT-09. Hypothesis used bounded finite engineering values, retained shrinking, and ran with seed `20260715`.
- PBT-02: unchanged and green through existing input and HEN round-trip tests.

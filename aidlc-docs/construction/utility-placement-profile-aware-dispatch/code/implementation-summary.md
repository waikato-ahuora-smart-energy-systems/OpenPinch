# Utility Placement Profile-Aware Dispatch Implementation Summary

## Outcome

Utility placement now optimizes period-specific hot and cold duty splits with
the shared utility temperatures. Detached targeting clips each requested duty
to profile availability and caller capacity, then assigns only the residual to
`HU` or `CU` fallback.

The optimizer uses normalized `[0, 1]` coordinates with structural temperature
ordering. Backend evaluation, canonical replay, alternative ordering, and best
candidate selection use the same entropy-plus-fallback scalar. Entropy is
normalized by a context-derived reference in `kW/K`, and termination evidence
includes canonical replay evaluations.

## TDD evidence

- RED: a two-level generated model exposed only temperature coordinates and no
  period duty coordinates.
- GREEN: dispatch encode/decode, allocation replay, ranking, scaling,
  accounting, Process/GCC, and Site/TSP regressions pass.
- Property evidence: stick-breaking duties are non-negative, conserve each
  period/side residual, and round-trip within the declared duty tolerance.
- Focused result: 220 utility-placement tests passed.
- Complete result: 2,445 tests passed with 4 expected skips.

## Extension compliance

- Property-Based Testing: compliant for round trips, conservation invariants,
  domain strategies, shrinking, and example regressions.
- Security Baseline: N/A because it is disabled and no external boundary was
  introduced.
- Resiliency Baseline: N/A because it is disabled and no service or deployment
  boundary was introduced.

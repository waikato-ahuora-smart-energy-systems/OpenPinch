# Utility Placement Exact-Target Replay Implementation Summary

## Outcome

Utility placement now optimizes utility temperatures and sensible spans only.
For every candidate and period, an application-owned adapter installs the
candidate utilities in a detached `PinchProblem` and invokes the ordinary
hierarchy-aware target workflow. That exact replay owns the named duties,
fallback duties, target totals, Total Site matching, and balanced-composite
snapshot used for entropy generation.

The returned case retains the utility temperatures and capacity metadata needed
by ordinary targeting. Direct, Total Site, and indirect retargeting reproduce
the optimizer evidence by utility name for Process, Site, Community, and Region
scopes without mutating the source case.

## TDD evidence

- RED: Process and Total Site regressions demonstrated that optimizer evidence
  and ordinary retargeting allocated different duties.
- GREEN: exact detached replay produces candidate-local allocation and entropy
  inputs; ordinary retargeting matches the selected evidence for every tested
  hierarchy and period.
- Independent duty-fraction coordinates and stick-breaking dispatch were
  removed because they could not be represented by a normal returned case.
- Property, focused analysis, application, architecture, and bounded-runtime
  gates pass.
- The complete solver-enabled suite passes 2,448 tests with 4 expected skips.
- Notebook 19 executes with two isothermal and two genuinely sensible levels,
  shows Process GCC and Site Total Site Profile outputs, and presents
  optimizer-versus-retarget comparison tables without tests or assertions.

## Extension compliance

- Property-Based Testing: compliant for codec round trips, temperature ordering,
  allocation conservation, detached replay, hierarchy equivalence, and example
  regressions.
- Security Baseline: N/A because it is disabled and no external boundary was
  introduced.
- Resiliency Baseline: N/A because it is disabled and no service or deployment
  boundary was introduced.

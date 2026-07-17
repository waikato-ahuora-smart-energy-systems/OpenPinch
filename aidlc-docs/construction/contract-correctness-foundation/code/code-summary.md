# Unit 1 Code Generation Summary

## Outcome

The contract and correctness foundation is complete. Multi-period aggregation
now treats a partially absent pinch temperature as an unavailable diagnostic
while retaining strict failure for partially absent required report metrics.
The package-root and target-method manifests establish the closed public
vocabulary used by the later accessor units.

## Application Changes

- `OpenPinch/application/_problem/periods/aggregation.py` adds an explicit
  optional-diagnostic policy to weighted report-value aggregation.
- Cold and hot pinch temperatures use that policy; required metrics such as
  exchanger area continue to reject partial absence.

## Test Changes

- Added an example regression contrasting optional pinch diagnostics with
  required metrics.
- Added a centralized strategy for valid, aligned two-period `TargetOutput`
  records.
- Added generated JSON round-trip, ordering, range, non-mutation, weight-scale,
  and optional-diagnostic properties.
- Added exact package-root exports and the planned descriptive target-method
  vocabulary as a shared contract foundation.

## Verification Evidence

- Focused pytest with Hypothesis seed `20260715`: 16 passed.
- Ruff lint: passed for all changed Unit 1 Python files.
- Ruff format check: passed for all changed Unit 1 Python files.
- Git patch hygiene: passed.

## Extension Compliance

- PBT-02: compliant through generated `TargetOutput` JSON round trips.
- PBT-03: compliant through range, ordering, scale, and non-mutation properties.
- PBT-07: compliant through `tests/strategies/period_outputs.py`.
- PBT-08: compliant through normal shrinking and the fixed verification seed.
- PBT-09: compliant through the existing Hypothesis and pytest integration.
- Security Baseline: N/A because the extension is disabled.
- Resiliency Baseline: N/A because the extension is disabled.

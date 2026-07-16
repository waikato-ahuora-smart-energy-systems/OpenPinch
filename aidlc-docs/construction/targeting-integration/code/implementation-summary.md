# Targeting and Integration Implementation Summary

## Outcome

Shared targeting calculations now expand piecewise thermal segments without changing physical stream identity, and heat-pump/refrigeration and MVR builders emit one segmented parent per physical duty or stage.

## Delivered

- Segment-expanded problem-table interval thermodynamics and parent-deduplicated stream counts.
- Segment-local HTC use in area and capital targeting inputs.
- One-parent condenser, evaporator, gas-cooler, Brayton, and MVR profile builders.
- One parent per process/direct MVR stage and one zone membership per parent.
- Stable multiperiod alignment on the union cumulative-duty-fraction grid.
- Regression assertions for parent equipment counts, segment continuity, target parity, and stable multiperiod identities.

## Verification

The focused domain, targeting, HPR, Brayton, process-component, and direct-MVR suite passed 187 tests. Ruff checks passed for all touched integration code and tests.

## Extension Compliance

- Security Baseline: disabled; not enforced.
- Resiliency Baseline: disabled; not enforced.
- Property-Based Testing (Partial): compliant through the shared segmented-stream strategy and invariant properties; integration behavior is additionally covered by deterministic thermodynamic fixtures.

## Pre-Release Summary Isolation and HPR Economics

Multiperiod summaries now capture one baseline zone and replay every period on
a fresh deep copy. The exact original zone object, cached target output,
target-run specification, and recording state are restored in `finally`, so
successful and exceptional replay cannot leak period mutations into the live
problem or another period.

Shared simulated-HPR candidate ranking now uses weighted operating cost plus
weighted feasibility penalty plus maximum annualized capital cost. Backends
without any cost breakdown retain weighted backend `obj` scoring. A partial
cost breakdown is rejected rather than silently mixing incompatible policies.

Public weighted HPR results continue to weight operating quantities and
operating cost. Total, annualized, compressor, and heat-exchanger capital fields
use their maximum period value, and total annualized cost is recomputed from
weighted operating cost plus maximum annualized capital. Non-HPR fields retain
their previous weighted-average behavior.

Verification for this unit:

- The regression baseline produced five expected failures and seventeen passes;
  all twenty-two tests passed after implementation.
- Expanded HPR/reporting and public problem/workspace/export/notebook groups
  passed 228 and 156 tests respectively.
- The complete seeded non-solver suite passed 2,004 tests with four solver
  cases deselected; the solver suite passed three with one intentional skip.
- Total line coverage was 98% across 23,201 statements, above the 95% gate.
- Repository Ruff lint, changed-file formatting, notebook JSON parsing,
  warning-free Sphinx, wheel/sdist packaging, and patch hygiene passed.

Extension compliance remains unchanged: Security and Resiliency are disabled,
and Property-Based Testing Partial remains compliant through the existing
generated domain invariants plus deterministic state-isolation and economic
aggregation regressions.

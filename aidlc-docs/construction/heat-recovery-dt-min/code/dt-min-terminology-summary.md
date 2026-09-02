# Heat-Recovery `dt_min` Terminology Summary

## Delivered contract

The inverse process-targeting API now uses one consistent `dt_min` naming
family:

- `problem.target.heat_recovery_dt_min(...)`
- `problem.target.all_periods.heat_recovery_dt_min(...)`
- active-workspace and ordered-case-batch mirrors
- `HeatRecoveryDtMinResult` and `HeatRecoveryDtMinStatus`
- result field `dt_min`
- `HeatRecoveryDtMinSolution`, `solve_heat_recovery_dt_min(...)`, and matching
  application orchestration functions and modules

The superseded service method is not retained as an alias and is explicitly
covered by the closed API test. Established exchanger-network
`approach_temperature` settings are unrelated contracts and remain unchanged.

## Updated surfaces

The rename includes analytical, application, property, architecture, package
inventory, notebook, documentation, and installed-artifact tests. Generated
notebooks 02 and 06, tutorial ownership data, the dedicated RTD guide,
fundamentals, public API pages, service references, navigation, overview, and
release notes all use `dt_min` as the primary vocabulary.

The threshold-limit behavior remains unchanged: a threshold problem returns
the greatest positive global `dt_min` that retains maximum recovery.

## Verification

- Focused inverse suite: 47 passed.
- Notebook and RTD gate: 52 passed, 3 expected skips.
- Warning-strict Sphinx: passed.
- Full configured suite: 2,584 passed, 4 skipped.
- Exact non-solver coverage selection: 2,581 passed, 3 skipped, 4 deselected.
- Branch coverage: 96 percent; the 95 percent gate passed.
- Wheel and source distribution: built successfully.
- Isolated Python 3.14 installed-wheel smoke: passed.
- Ruff lint, feature-file formatting, and patch hygiene: passed.

Security and Resiliency extensions remained disabled. Property-Based Testing
rules PBT-01 through PBT-05 and PBT-07 through PBT-10 remain satisfied; PBT-06
is N/A because the solver owns no persistent mutable state.

# Unit 2 Code Generation Summary

The callable target and configuration-selected analysis plan were removed.
`PinchProblem.target` now exposes only descriptive heat-integration, HPR,
cogeneration, exergy, energy-transfer, area/cost, and mirrored all-period
methods. Named invocation arguments override advanced options and stored
numerical configuration ephemerally; configuration no longer selects a core
method. Observation, plotting, reports, dashboard, and Excel export require
existing results and do not solve.

Target, period, HPR, workspace, reporting, generated precedence, structural
traversal, and public-contract verification passed. The combined refreshed
application checkpoint passed 137 tests before Unit 3, and the expanded Unit 3
checkpoint subsequently passed 232 tests. Focused Ruff lint and format gates
pass. Intentional breaks include `problem.target()`, `target_all_periods`, old
direct/indirect HPR spellings, target selector configuration, redundant JSON
constructors, mutable export destination state, and implicit observation solves.

Partial PBT rules PBT-02, PBT-03, PBT-07, PBT-08, and PBT-09 are compliant via
central strategies, fixed seed policy, normal shrinking, precedence properties,
and period ordering/non-mutation properties. Security and Resiliency are
disabled and N/A.

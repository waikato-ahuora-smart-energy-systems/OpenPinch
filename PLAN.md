# Refactor `power_cogeneration_service`

## Summary
Simplify cogeneration targeting into a predictable “select candidate -> ensure that exact target exists for the requested state -> apply turbine analysis” flow.

Keep the public API shape the same, but change the implicit default behavior of `problem.target.cogeneration()` so it prefers:

`TS -> IHP -> IR -> DHP -> DR -> DI`

This matches your preference for Total Site first, then HPR targets, then Direct Integration.

## Key Changes
- In [services_entry.py](/Users/timothyw/Github_Local/OpenPinch/OpenPinch/services/services_entry.py), split cogeneration orchestration into small internal helpers:
  - one helper to validate/normalize `base_target_type`
  - one helper to produce the candidate order for the current call
  - one helper to ensure a specific candidate target exists for the requested state by calling the mapped prerequisite service
- Make `power_cogeneration_service()` operate on one selected candidate at a time instead of looping over a mixed priority list and mutating whichever target happens to exist.
- Treat fallback only as a “target unavailable” condition:
  - if `TS` cannot be produced for the zone/state, continue to the next candidate
  - if a candidate exists and cogeneration finds no viable turbine stage, keep that target and return it with zero/no work result rather than falling through to another base target
- Keep explicit `base_target_type` as the override path:
  - explicit type disables fallback
  - unsupported explicit types raise `ValueError`
  - explicit supported type that cannot be produced for the requested state raises a clear runtime error
- Fix the accessor/return mismatch by making cogeneration return the actual target family selected by the service, not a hardcoded `DI` target:
  - add a dedicated internal cogeneration execution path in `PinchProblem` / the accessor flow
  - have the service persist the selected cogeneration target type on the zone for that execution
  - return `zone.targets[selected_type]` after targeting completes
- In [power_cogeneration_analysis.py](/Users/timothyw/Github_Local/OpenPinch/OpenPinch/services/power_cogeneration_analysis/power_cogeneration_analysis.py), rename the conceptual input from `zone` to `target` in type hints/docstrings and document that cogeneration operates on compatible target objects carrying `config`, `hot_utilities`, `work_target`, and `turbine_efficiency_target`.
- Update accessor docstrings and any user-facing notes so the implicit default order is clear.

## Public API / Behavior
- No signature change to `problem.target.cogeneration(...)`.
- `options["base_target_type"]` remains the public override.
- Behavioral change:
  calls without `base_target_type` no longer imply `DI`; they now resolve by `TS -> IHP -> IR -> DHP -> DR -> DI`.
- Compatible target families for cogeneration become:
  `TS`, `IHP`, `IR`, `DHP`, `DR`, `DI`.

## Test Plan
- Add a regression where both `TS` and `DI` exist and implicit cogeneration mutates and returns `TS`, not `DI`.
- Add fallback-order tests:
  - `TS` unavailable -> falls to `IHP`
  - no `TS`/HPR target available -> falls to `DI`
- Add explicit override tests:
  - explicit `TS`, `IHP`, and `DI` each target exactly that family
  - unsupported explicit type raises `ValueError`
  - explicit supported type that cannot be produced raises a clear error
- Add state-sensitive tests proving the selected candidate is refreshed when its cached target was solved for a different state.
- Add a regression that “no viable turbine stage” does not trigger fallback to another base target.
- Keep the existing analysis-layer test that `get_power_cogeneration_above_pinch()` accepts HPR targets.

## Assumptions
- Within the HPR group, preserve the current implicit priority order: `IHP -> IR -> DHP -> DR`.
- `TZ`, `RT`, `TL`, and `ET` remain unsupported for cogeneration.
- Cogeneration continues to mutate the selected base target in place via `work_target` and `turbine_efficiency_target`; it does not create a new dedicated cogeneration target model.

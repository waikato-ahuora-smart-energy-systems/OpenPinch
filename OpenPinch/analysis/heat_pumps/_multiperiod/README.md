# Multiperiod HPR Targeting

## Overview
Private helpers that solve one shared heat pump or refrigeration (HPR) design vector against several operating periods and return a single `HeatPumpTargetOutputs`. `service.py` in the parent package calls this directory; it reuses the cycle-specific objectives from `../targeting`, the search and translation functions in `../optimisation_adapter`, and the preflight functions in `../performance_maps`.

## Interface
Imported by `service.py`:
- `get_multiperiod_hpr_targets` (`execution.py`): build `MultiPeriodHPRTargetInputs` from prepared cases, run the shared-vector search and translate the winner to `HeatPumpTargetOutputs`.
- `build_multiperiod_hpr_cases` (`preparation.py`): compute base targets and aligned problem tables per period and return `_PreparedHPRPeriodCase` records (periods with no HPR load are dropped).
- `idle_period_weight` (`preparation.py`): total weight of the zone's periods that have no prepared case.
- `period_case_by_id`, `period_id_for_index` (`preparation.py`): look up a prepared case by id, or a period id from an index.

Internal to the directory:
- `solve_hpr_multiperiod_placement` (`execution.py`): search one shared vector across all period cases and return the first successful candidate, with its alternatives, as a translated result, or raise an HPR targeting error.
- `reject_penalised_periods` (`execution.py`): reject a shared design still penalised in any period, each period checked on its own result and inputs.
- `evaluate_multiperiod_candidate`, `selected_period_case` (`aggregation.py`): evaluate one design vector in every period and aggregate; resolve the selected period case.
- `get_multiperiod_hpr_optimisation_setup` (`setup.py`): return starting points, bounds, the period objective and, for vapour compression cycles, the stage lifts for a cycle type.
- `_PreparedHPRPeriodCase` (`state.py`): slotted dataclass holding period id, index, weight, solver case, base target and optimiser problem table.

## Invariants
- Periods are processed in the order of `zone.period_ids` (default `{"0": 0}`), and the aggregated `period_outputs` keep that order [aggregation.py:evaluate_multiperiod_candidate, preparation.py:_canonical_period_items].
- A period whose target load is below `tol` is excluded from the shared design; its weight is carried separately as the idle weight [preparation.py:build_multiperiod_hpr_cases, preparation.py:idle_period_weight].
- Period problem tables are merged onto one shared temperature grid by inserting missing intervals in place; preparation raises `ValueError` if the merged grids differ in length or by more than `tol`, or, in `build_multiperiod_hpr_cases`, if a period has no base target [preparation.py:_align_hpr_problem_tables] [preparation.py:build_multiperiod_hpr_cases].
- A shared design is "no heat pump" only if every period returns `ZERO_USEFUL_DUTY_REASON`; otherwise idle periods are re-evaluated in search mode and priced as an idle machine [aggregation.py:evaluate_multiperiod_candidate, aggregation.py:_as_idle_period].
- Any non-idle period failure fails the whole candidate with a reason naming the period [aggregation.py:_period_failure].
- For vapour compression cycles the setup also returns the stage lifts (`x, args -> lifts`). Each candidate is first decoded in every period without a floor; each stage's design lift is its largest lift in any period, and every period then decodes with `stage_lift_floor = HPR_TURNDOWN x design lift`, its condensers raised where a stage lifts less. There is no separate turndown penalty: the higher condensing temperature and worse COP are the cost. The floor is set on each period's `args` per candidate. [aggregation.py:_set_stage_lift_floors] [setup.py:get_multiperiod_hpr_optimisation_setup]
- The selected period is resolved by id first, then by index; absence raises `ValueError` [aggregation.py:selected_period_case].
- Candidates are tried in ranked order and the loop stops at the first zero-useful-duty candidate [execution.py:solve_hpr_multiperiod_placement].
- A shared design is rejected when any period's own feasibility penalty exceeds `PENALTY_TOLERANCE` of that period's objective, however light its weight; the failure reports that period's served and targeted duty. Alternatives are re-evaluated across every period and checked the same way. [execution.py:reject_penalised_periods] [execution.py:solve_hpr_multiperiod_placement]
- The TESPy backend is rejected for multiperiod inputs by `preflight_tespy_hpr_targeting` [execution.py:get_multiperiod_hpr_targets, ../performance_maps/targeting.py:preflight_tespy_hpr_targeting].
- Brayton raises `NotImplementedError`; vapour compression with MVR is heat-pump-only [setup.py:get_multiperiod_hpr_optimisation_setup].
- Parallel cycle types force `n_cond == n_evap` on every period case [setup.py:get_multiperiod_hpr_optimisation_setup].

## Side Effects
- Zone target dictionaries are temporarily extended while period base targets are computed and restored on exit [preparation.py:_preserved_zone_targets].
- `case.args` of each period case is mutated in place (`n_cond`, `n_evap`, `refrigerant_ls`; for VC+MVR a non-empty `mvr_fluid_ls` becomes `refrigerant_ls` before the fluid preflight and the Carnot seed) [setup.py:get_multiperiod_hpr_optimisation_setup].
- CoolProp is queried by the fluid and state preflights when the backend is `coolprop` [setup.py:get_multiperiod_hpr_optimisation_setup].
- Problem tables are deep-copied for direct targets [preparation.py:_optimizer_problem_table_for_hpr].
- `_align_hpr_problem_tables` mutates each period's optimiser problem table by inserting shared temperature intervals [preparation.py:_align_hpr_problem_tables].
- Setup runs a Carnot seed optimisation (`initialise_hpr_seed`) for cascade, parallel and VC+MVR types when `hpr.initialise_simulated_cycle` is true (the default), which is a full solve; a seed failure is ignored [setup.py:get_multiperiod_hpr_optimisation_setup].

## Verification
- Run: `uv run --no-sync pytest tests/analysis/heat_pumps/test_multiperiod_hpr.py`

# HPR Common Calculations

## Overview
Calculation modules shared by the heat-pump and refrigeration (HPR) cycle
models, the per-topology targeting modules and the service entry point in the
parent package. They cover background-profile preparation, optimisation vector
layout and encoding, load selection, costing and result assembly, the shared
multi-stage vapour-compression objective, and residual utility accounting.
`__init__.py` has only a docstring.

## Interface
- `construct_HPRTargetInputs`, `default_refrigeration_to_power_ratio` (`preprocessing.py`): build `HeatPumpTargetInputs` (shifted, simplified background profiles, ambient flags, cost and cycle settings) from a zone's cascades and `Configuration`; default refrigeration electricity ratio.
- `resolve_hpr_target_load` (`load_selection.py`): requested HPR duty from `config.hpr.load_mode` (`fraction`, `duty`, `period_values`), capped by the available cascade duty.
- `HPRoptVectorLayout`, `clip_to_bounds` (`layout.py`): named sections of a flat optimisation vector (`pack`, `unpack`, `build_bounds`) and clipping of seeds into bounds.
- Duty and temperature encoding (`encoding.py`): `decode_available_fractions`, `encode_available_fractions`, `limit_available_duty`, `allocate_stage_duties`, `require_stage_duty_allocation`, `DutyAllocation`, `StageDutyRequest`, `DutyAllocationRequest` (with `from_state`), and the `map_*_to_*_arr` pairs between optimisation fractions and temperatures, temperature differences or duties.
- Costing and results (`shared.py`): `HPRCostUnit`, `calc_hpr_capital_cost`, `calc_hpr_machine_capital_costs`, `calc_simulated_hpr_annualized_costs`, `hpr_penalty_cost_scale`, `refrigeration_allowance`, `evaluate_carnot_hpr_result`, `evaluate_vapour_hpr_result`, `post_solve_error`.
- Search-range helpers (`shared.py`): `condensing_temperature_search_range`, `cap_stage_condensing_temperatures`, `floor_stage_evaporating_temperatures`, `validate_vapour_hp_refrigerant_ls`, `is_negligible_useful_duty`, and the constants `SUBCRITICAL_CONDENSING_MARGIN_K`, `NEGLIGIBLE_USEFUL_DUTY_FRACTION`.
- Thermodynamic helpers (`shared.py`): `calc_carnot_heat_pump_cop`, `calc_carnot_heat_engine_eta`, `compute_entropic_mean_temperature`, and `satisfies_second_law` (public but not in `__all__`).
- `_evaluate_multi_vc_objective` (`multi_vc_objective.py`): shared objective for cascade and parallel vapour-compression targeting (parse vector, solve, build streams, account, attach CoolProp record).
- `value_at_index`, `to_kelvin`, `to_degrees_celsius`, `to_kilopascal` (`units.py`): period-indexed value and unit conversion.
- `postprocessing.py`: only private functions; `_get_hpr_residual_utility_summary` is the one the service imports, retargeting utilities against the residual cascade after the HPR.
- Subdirectories: [_shared](./_shared/README.md): private air, stream and plotting helpers behind `shared.py`.

## Invariants
- Background profiles are shifted by `hpr.dt_cont` (hot temperatures down, cold temperatures up) and extended by 10 K at each end; `Q_cool_max` is taken as minus the last hot enthalpy, so the hot enthalpy profile is expected to be negative. [preprocessing.py:construct_HPRTargetInputs] [preprocessing.py:_extend_profile_with_temperature_margin]
- A refrigerator searches only the coldest `Q_hpr_target` of the cooling, while the warmer rest is kept as `untargeted_cooling_streams`. [preprocessing.py:_split_cooling_profile_at_coldest_Q_target]
- A zero target raises `ValueError` for heat pumping and refrigeration profiles. [preprocessing.py:_get_reduced_bckgrd_cascade_till_Q_target]
- The default refrigeration price ratio is never below the cooling-water ratio. [preprocessing.py:default_refrigeration_to_power_ratio]
- Stage duties are decoded as fractions in [0, 1] of each stage's available duty, so no stage can request more than it can deliver. [encoding.py:decode_available_fractions]
- `HPRoptVectorLayout` section order is cond, evap, subcool, heat_split, cool_split, ihx, misc, and counts must be non-negative. [layout.py:HPRoptVectorLayout]
- The load is `Q_max * load_fraction` in `fraction` mode and is capped at the available duty `Q_max` in `duty` and `period_values` modes; an unknown mode, a missing `config` or a missing period key raises `ValueError`. [load_selection.py:resolve_hpr_target_load]
- Machine capital is `F_inst * C_eq * (Q_cap/MW)^exp * stage factor * temperature factor`, per machine, and zero for a machine with no capacity. [shared.py:calc_hpr_machine_capital_costs]
- A cascade stage whose work is at most `NEGLIGIBLE_STAGE_WORK_FRACTION` of the machine's work is not charged stage capital. [shared.py:_single_cost_unit]
- The feasibility penalty is `rho * sum(r + r^2)` with violations relative to `Q_hpr_target`, priced at the no-heat-pump annual cost, and never below 1 $/y. [shared.py:_cycle_penalty] [shared.py:hpr_penalty_cost_scale]
- Errors raised after a cycle has solved are re-raised, not ranked as infeasible candidates. [multi_vc_objective.py:_evaluate_multi_vc_objective] [shared.py:post_solve_error]
- Ambient air and cooling water are applied through `cascade_with_air` on the condenser and evaporator cascades; cooling water applies on the evaporator side only. [shared.py:evaluate_vapour_hpr_result]
- `calc_carnot_heat_pump_cop` and `calc_carnot_heat_engine_eta` take absolute temperatures (K); `satisfies_second_law` takes degC and adds 273.15 internally, and `compute_entropic_mean_temperature` takes degC by default and returns K. [shared.py:calc_carnot_heat_pump_cop] [shared.py:satisfies_second_law]
- Residual loads below `max(tol, 1e-7 * largest)` are treated as rounding noise and set to zero; duplicate residual temperatures with conflicting values raise `ValueError`. [postprocessing.py:_residual_noise_floor] [postprocessing.py:_deduplicate_residual_profile_rows]
- `common` imports the parent's `optimisation_adapter` and `performance_maps.targeting` for accounting and backend names. [shared.py] [preprocessing.py]

## Side Effects
- Imports CoolProp at module load; calls `PropsSI` for `Tmin` and `Tcrit` and `critical_point` for critical temperatures. [shared.py:floor_stage_evaporating_temperatures] [shared.py:validate_vapour_hp_refrigerant_ls]
- `refrigeration_allowance` caches its result by setting an attribute on `args`. [shared.py:refrigeration_allowance]
- `_get_hpr_residual_load_profiles` writes the post-HPR profile columns into the problem table passed in. [postprocessing.py:_get_hpr_residual_load_profiles]
- `construct_HPRTargetInputs` copies its array arguments before modifying them. [preprocessing.py:construct_HPRTargetInputs]
- Optional Plotly use through `_shared.plotting` when `debug` is set on a final evaluation. [shared.py:evaluate_vapour_hpr_result]

## Verification
- Run: `uv run --no-sync pytest tests/analysis/heat_pumps/test_shared.py tests/analysis/heat_pumps/test_encoding.py tests/analysis/heat_pumps/test_layout.py tests/analysis/heat_pumps/test_preprocessing.py`
- `postprocessing.py` is covered by `tests/analysis/heat_pumps/test_hpr_residual_regressions.py`; `load_selection.py` by `tests/analysis/heat_pumps/test_targeting.py`. `multi_vc_objective.py` and `units.py` have no direct test; they are exercised through the cascade and parallel vapour-compression tests.

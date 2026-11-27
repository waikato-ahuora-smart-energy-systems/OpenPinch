# Pinch Targeting

## Overview
The deterministic core of pinch analysis: the problem table cascade, grand composite curve variants, multi-utility targeting, direct integration of one zone, indirect (Total Site) integration across subzones, area and capital cost targets, and an inverse solver for a global heat-recovery dt_min. The services in the sibling `energy_transfer`, `exergy` and `power` packages reuse `context.py` for period matching and enrichment set-up. The `heat_pumps`, `utility_placement` and `orc` packages import from the cascade, grand composite and utility modules.

## Interface
- `compute_direct_integration_targets(zone, args, *, prepared_profile)` (`direct.py`): builds a `DirectIntegrationTarget` for a zone (problem tables, utility targets, optional balanced curves, area and cost).
- `build_area_cost_target_data`, `should_update_balanced_composite_curves` (`direct.py`): area, unit count, capital and total cost fields; balanced-curve switch.
- `compute_subzone_aggregate_target`, `compute_indirect_integration_targets` (`indirect.py`): sum subzone Direct targets; build the Total Site target from net stream profiles.
- `get_process_heat_cascade`, `get_utility_heat_cascade`, `create_problem_table_with_t_int`, `problem_table_algorithm`, `get_heat_recovery_target_from_pt`, `set_zonal_targets` (`cascade.py`): problem table construction and zonal targets.
- `get_additional_GCCs`, `get_GCC_without_pockets`, `get_GCC_with_partial_pockets`, `get_GCC_with_vertical_heat_transfer`, `get_GCC_needing_utility`, `get_GGC_pockets`, `get_seperated_gcc_heat_load_profiles` (`grand_composite.py`): GCC variants and the hot/cold net load profiles.
- `get_utility_targets`, `target_utilities_for_load_profiles`, `mark_default_utility`, `is_default_utility` (`utilities.py`): assign duties to multiple utilities.
- `get_balanced_CC`, `get_area_targets`, `get_min_number_hx`, `get_capital_cost_targets` (`area_cost.py`): balanced composites, area, minimum unit count, capital cost.
- `get_temperature_driving_forces` (`temperature_driving_force.py`): interval driving forces between two balanced composites.
- `solve_heat_recovery_dt_min`, `evaluate_process_heat_recovery`, `HeatRecoveryDtMinSolution`, `HeatRecoveryLimitError` (`heat_recovery_dt_min.py`): invert recovery to a global dt_min by bisection.
- `allocate_residual_utilities`, `compute_residual_utility_target` (`residual.py`): utilities on a frozen heat pump residual.
- `record_selected_period`, `target_matches_requested_period`, `apply_zone_config_overrides`, `format_selected_period_suffix`, `normalize_base_target_type`, `prepare_enrichment_run`, `clone_target_with_zone_settings` (`context.py`): shared service helpers.
- Subdirectories: none.

## Invariants
- Problem tables run from high to low shifted temperature; in the shifted-temperature table `H_NET` is `-cumsum(dT * CP_NET)` shifted so its minimum is zero, so the first row is the hot utility target and the last the cold utility target. The real-temperature table is re-shifted to the shifted table's heat recovery, so its `H_NET` minimum is the gap between the two heat recoveries. [cascade.py:_problem_table_algorithm] [cascade.py:set_zonal_targets] [cascade.py:_shift_pt_to_set_heat_recovery]
- Heat recovery is `H_HOT[0] - H_NET[-1]`. [cascade.py:get_heat_recovery_target_from_pt]
- Ordinary targeting uses the canonical temperature grid with an interval tolerance of `10 * tol`; the inverse dt_min path uses exact temperatures and zero tolerances. [cascade.py:problem_table_algorithm] [cascade.py:_get_precise_process_heat_cascade]
- Runtime `args` may contain only `period_idx`, `period_id`, `base_target_type`, `_calculate_area_cost`, `_prepared_direct_profiles`, `maximum_iterations`, `maximum_evaluations` and `simulation_backend`; other keys raise `ValueError`. [context.py:apply_zone_config_overrides]
- An existing target matches a request only if its period matches and (when a `config` is passed) no `THERMAL_`, `DIRECT_`, `INPUT_UNIT_`, `OUTPUT_UNIT_` or `COSTING_` setting has changed. [context.py:target_matches_requested_period]
- Hot utilities are assigned coldest first and cold utilities hottest first, by shifted level (real level when `is_real_temperatures` is set) in the selected period; default utilities added during input preparation always go last. [utilities.py:_calculate_assigned_utility_duties]
- No utility available when the profile needs one raises `ValueError`; unmet demand after capping only warns. [utilities.py:_calculate_utility_duties_for_load_profiles] [utilities.py:_warn_on_unmet_utility_demand]
- Balanced composite curves must have equal hot and cold duty, otherwise `ValueError`. [area_cost.py:get_area_targets] [temperature_driving_force.py:get_temperature_driving_forces]
- A zero or negative approach makes `get_area_targets` warn and return infinity when such an interval carries duty (zero-duty intervals are ignored), rather than raise. [area_cost.py:get_area_targets]
- `dt_min` and requested recovery must be finite and non-negative; a request above the zero-dt_min limit raises `HeatRecoveryLimitError`; bisection stops within 1e-6, and needing more than 100 iterations raises `RuntimeError`. [heat_recovery_dt_min.py:solve_heat_recovery_dt_min] [heat_recovery_dt_min.py:DT_MIN_TOLERANCE]
- Indirect targeting needs a `SubzoneAggregateTarget` on the zone and, when the zone has subzones, a Direct target on each immediate subzone (a missing one raises `KeyError`); a zone without subzones uses its own net streams; it returns `None` when there are no net streams. [indirect.py:compute_indirect_integration_targets]
- Utility totals in the Total Site target come from cascading use and generation by temperature interval, with same-name levels matched first. [indirect.py:_target_site_utilities]

## Side Effects
- `compute_direct_integration_targets` sets `zone.net_hot_streams` and `zone.net_cold_streams`, and writes utility duties into `zone.hot_utilities` and `zone.cold_utilities`. [direct.py:compute_direct_integration_targets]
- `compute_indirect_integration_targets` sets `zone.subzone_net_hot_streams` and `zone.subzone_net_cold_streams`. [indirect.py:_reconstruct_subzone_direct_profiles]
- `get_utility_targets` mutates its problem tables and, when `is_direct_integration` is true, writes duties into the utility streams passed in. [utilities.py:get_utility_targets] [utilities.py:_apply_utility_duties]
- `get_area_targets` and `_warn_on_unmet_utility_demand` emit `UserWarning`. [area_cost.py:get_area_targets] [utilities.py:_warn_on_unmet_utility_demand]
- `record_selected_period` writes `zone._selected_period_id` and `zone._selected_period_idx`. [context.py:record_selected_period]
- `mark_default_utility` sets an attribute on a stream. [utilities.py:mark_default_utility]
- The dt_min solver works on deep copies of the streams, so the caller's streams are unchanged. [heat_recovery_dt_min.py:solve_heat_recovery_dt_min]

## Verification
- Run: `uv run --no-sync pytest tests/analysis/test_direct_targeting.py tests/analysis/test_total_site_targeting.py tests/analysis/test_utility_targeting.py tests/analysis/test_problem_table.py tests/analysis/test_gcc_manipulation.py tests/analysis/test_capital_cost_and_area_targeting.py tests/analysis/test_temperature_driving_force.py tests/analysis/test_heat_recovery_dt_min.py`
